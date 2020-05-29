%% Main script for calling mouse data simulation network
% 2019-08-14 plotting now handles multiple varied parameters
% 2019-09-11 moved plotting to a separate script plotPerfGrid.m
%            mouseNetwork returns both R and C performance
%            now plot both R and C performance grids
%
% to do:
%   add ability to adjust RC netcon in main code
clearvars;
close all;

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('dynasim'))

ICdir = ['MiceSpatialGrids/' 'ICStim/Mouse/s30_sg0.5_ml0.01_20200424-145616'];
ICdirPath = [ICdir filesep];
ICstruc = dir([ICdirPath '*.mat']);
if isempty(ICstruc), error('empty data directory'); end

% specify the set or subset of configurations to run
% subz = find(contains({ICstruc.name},'m0.mat')); % sXm0 (target only) cases
subz = 1:length(ICstruc);
%% varied parameters
varies(1).conxn = '(IC->IC)';
varies(1).param = 'trial';
varies(1).range = 1:20;

varies(end+1).conxn = 'C';
varies(end).param = 'noise';
varies(end).range = 0.03;%:0.01:0.05;

variedParam = 'CNoise';
% variedParam = 'S-R_gsyn';
% varies(end+1).conxn = '(IC->R)';
% varies(end).param = 'gSYN';
% varies(end).range = .2; %0.15:0.005:0.19;

%% netcons
nCells = 4; %synchronise this variable with mouse_network

% x-channel inhibition
% irNetcon = diag(ones(1,nCells))*0.1;
irNetcon = zeros(nCells);
% irNetcon(2,1) = 1;
% irNetcon(3,1) = 1;
% irNetcon(4,1) = 1;
% irNetcon(2,4) = 1;

% sharpening
srNetcon = diag(ones(1,nCells));

% may need to make rcNetcon have variable weights
rcNetcon = ones(4,1)*.5; 

netCons.irNetcon = irNetcon;
netCons.srNetcon = srNetcon;
netCons.rcNetcon = rcNetcon;
%% Initialize variables
plot_rasters = 1;

nvaried = {varies(2:end).range};
nvaried = prod(cellfun(@length,nvaried));
diagConfigs = [6,12,18,24];
datetime=datestr(now,'yyyymmdd-HHMMSS');

set(0, 'DefaultFigureVisible', 'off')
h = figure('Position',[50,50,850,690]);
for z = subz %1:length(ICstruc)
    % restructure IC spikes
    load([ICdirPath ICstruc(z).name],'t_spiketimes');
    temp = cellfun(@max,t_spiketimes,'UniformOutput',false);
    tmax = max([temp{:}]);
    spks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
    for j = 1:size(t_spiketimes,1) %trials [1:10]
        for k = 1:size(t_spiketimes,2) %neurons [(1:4),(1:4)]
            if k < 5 %song 1
                spks(j,k,round(t_spiketimes{j,k})) = 1;
            else
                spks(j+10,k-4,round(t_spiketimes{j,k})) = 1;
            end
        end
    end

    % save spk file
    spatialConfig = strsplit(ICstruc(z).name,'.');
    study_dir = fullfile(pwd, 'run', datetime, spatialConfig{1});
    if exist(study_dir, 'dir')
      rmdir(study_dir, 's');
    end
    mkdir(fullfile(study_dir, 'solve'));
    save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');

    % call network
    h.Name = ICstruc(z).name;
    time_end = size(spks,3);
    [data(z).perf, data(z).fr, data(z).annot] = mouse_network(study_dir,time_end,varies,netCons,plot_rasters);
    data(z).name = ICstruc(z).name;
end
save([pwd filesep 'run' filesep, datetime filesep 'summary_results.mat'],'data')

% figure;
% for ii = 1:4
%     subplot(1,4,ii)
%     plotSpikeRasterFs(flipud(logical(squeeze(spks(:,ii,:)))), 'PlotType','vertline');
%     xlim([0 2000])
% end

%% performance grids
% performance vector has dimensions [numSpatialChan,nvaried]
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

temp = {data.name};
temp(cellfun('isempty',temp)) = {'empty'}; %label empty content
targetIdx = find(contains(temp,'m0'));
maskerIdx = find(contains(temp,'s0'));
% mixedIdx = setdiff(1:length(temp),[targetIdx,maskerIdx]);
mixedIdx = find(~contains(temp,'m0') & ~contains(temp,'s0') & ~contains(temp,'empty'));
numNeuronTypes = 4;

h = figure('position',[200 200 1600 600]);
for vv = 1:nvaried
    if ~isempty(mixedIdx)
        % mixed config cases
        for i = 1:length(mixedIdx)
            perf.IC(i,:) = data(mixedIdx(i)).perf.IC(:,vv);
            perf.R(i,:) = data(mixedIdx(i)).perf.R(:,vv);
            perf.C(i) = data(mixedIdx(i)).perf.C(vv);
            fr.IC(i,:) = data(mixedIdx(i)).fr.IC(:,vv);
            fr.R(i,:) = data(mixedIdx(i)).fr.R(:,vv);
            fr.C(i) = data(mixedIdx(i)).fr.C(vv);
        end

        % relay neurons
        order = [1,2,3,4];
        for nn = 1:numNeuronTypes
            subplot(2,5,order(nn))
            plotPerfGrid(perf.R(:,nn),fr.R(:,nn),neurons(nn));
            subplot(2,5,order(nn)+5)
            plotPerfGrid(perf.IC(:,nn),fr.IC(:,nn),[]);
        end
        subplot(2,5,1);
        ylabel({'R neurons',' ',' '})
        subplot(2,5,6)
        xticklabels({'-90','0','45','90'})
        yticklabels(fliplr({'-90','0','45','90'}))
        xlabel('Song Location')
        ylabel({'IC neurons','Masker Location'})
        % C neuron
        subplot('Position',[0.8 0.15 0.15 0.35])
        plotPerfGrid(perf.C',fr.C',[]);
    end
    
    % C neuron; target or masker only cases
    perf.CT = zeros(1,4);
    perf.CM = zeros(1,4);
    fr.CT = zeros(1,4);
    fr.CM = zeros(1,4);
    if ~isempty(targetIdx)
        for i = 1:length(targetIdx)
            perf.CT(i) = data(targetIdx(i)).perf.C(vv);
            fr.CT(i) = data(targetIdx(i)).fr.C(vv);
        end
    end    
    if ~isempty(maskerIdx)
        for i = 1:length(targetIdx)
            perf.CM(i) = data(maskerIdx(i)).perf.C(vv);
            fr.CM(i) = data(maskerIdx(i)).fr.C(vv);
        end
    end    
    subplot('Position',[0.8 0.6 0.15 0.2])
    plotPerfGrid([perf.CT;perf.CM],[fr.CT;fr.CM],'Cortical');
    
    % simulation info
    annotation('textbox',[.8 .85 .15 .2],...
           'string',data(z).annot(vv,3:end),...
           'FitBoxToText','on',...
           'LineStyle','none')

    % save grid
    Dirparts = strsplit(study_dir, filesep);
    DirPart = fullfile(Dirparts{1:end-1});
    saveas(gca,[filesep DirPart filesep 'SpatialGrid vary ' variedParam num2str(varies(end).range(vv),'%0.2f') '.tiff'])
    clf
end
set(0, 'DefaultFigureVisible', 'on')
