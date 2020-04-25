%% Main script for calling mouse data simulation network
% 2019-08-14 plotting now handles multiple varied parameters
% 2019-09-11 moved plotting to a separate script plotPerfGrid.m
%            mouseNetwork returns both R and C performance
%            now plot both R and C performance grids
%
% to do:
%   add ability to adjust RC netcon in main code
clearvars -except saveName;
close all;

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('dynasim'))

researchDrive = 'MiceSpatialGrids/';

ICdir = [researchDrive,'ICStim/Mouse/',saveName];
% ICdir = 'ICSimStim/mouse/v2/145638_s30';
ICdirPath = [ICdir filesep];
ICstruc = dir([ICdirPath '*.mat']);
if isempty(ICstruc), error('empty data directory'); end
%% varied parameters
varies(1).conxn = '(IC->IC)';
varies(1).param = 'trial';
varies(1).range = 1:20;

varies(end+1).conxn = 'C';
varies(end).param = 'noise';
varies(end).range = 0.03;%:0.01:0.05;

varies(end+1).conxn = '(S->R)';
varies(end).param = 'gSYN';
varies(end).range = 0.19:0.01:0.22; %0.15:0.005:0.19;

variedParam = 'S-R_gsyn';
% varies(end+1).conxn = '(IC->R)';
% varies(end).param = 'gSYN';
% varies(end).range = .2; %0.15:0.005:0.19;

%% netcons
nCells = 4; %synchronise this variable with mouse_network

% irNetcon = diag(ones(1,nCells))*0.1;
irNetcon = zeros(nCells);
% irNetcon(2,1) = 1;
% irNetcon(3,1) = 1;
% irNetcon(4,1) = 1;
% irNetcon(2,4) = 1;

srNetcon = diag(ones(1,nCells));
% srNetcon = zeros(nCells);

rcNetcon = zeros(4,1); %add this as input to mouse_network
% make rnNetcon have variable weights (instead of zeros)

netCons.irNetcon = irNetcon;
netCons.srNetcon = srNetcon;
netcons.rcNetcon = rcNetcon;
%% Initialize variables
plot_rasters = 1;

nvaried = {varies(2:end).range};
nvaried = prod(cellfun(@length,nvaried));
diagConfigs = [6,12,18,24];
datetime=datestr(now,'yyyymmdd-HHMMSS');

set(0, 'DefaultFigureVisible', 'off')
h = figure('Position',[50,50,850,690]);

subz = find(contains({ICstruc.name},'m0.mat')); % sXm0 (target only) cases
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
    study_dir = fullfile(pwd, 'run', datetime, filesep, spatialConfig{1});
    if exist(study_dir, 'dir')
      rmdir(study_dir, 's');
    end
    mkdir(fullfile(study_dir, 'solve'));
    save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');
    addpath(fullfile(pwd, 'run', datetime))
    addpath(fullfile(study_dir,'solve'));

    % call network
    h.Name = ICstruc(z).name;
    time_end = size(spks,3);
    [data(z).perf, data(z).annot] = mouse_network(study_dir,time_end-1,varies,netCons,plot_rasters);
    data(z).name = ICstruc(z).name;
end

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
textColorThresh = 70;
numSpatialChan = 4;

h = figure;
for vv = 1:nvaried
    if ~isempty(mixedIdx)
        % mixed config cases
        for i = 1:length(mixedIdx)
            perf.R(i,:) = data(mixedIdx(i)).perf.R(:,vv);
            perf.C(i) = data(mixedIdx(i)).perf.C(vv);
        end
        figure('position',[200 200 1200 600]);

        % relay neurons
        order = [1,2,4,5];
        for nn = 1:numSpatialChan
            subplot(2,3,order(nn))
            plotPerfGrid(perf.R(:,nn),neurons(nn),textColorThresh);
        end

        % C neuron
        subplot('Position',[0.7 0.15 0.2 0.4])
        plotPerfGrid(perf.C',[],textColorThresh);
    end
    
    % C neuron; target or masker only cases
    perf.CT = zeros(1,4);
    perf.CM = zeros(1,4);
    if ~isempty(targetIdx)
        for i = 1:length(targetIdx)
            perf.CT(i) = data(targetIdx(i)).perf.C(vv);
        end
    end    
    if ~isempty(maskerIdx)
        for i = 1:length(targetIdx)
            perf.CM(i) = data(maskerIdx(i)).perf.C(vv);
        end
    end    
    subplot('Position',[0.7 0.6 0.2 0.2])
    plotPerfGrid([perf.CT;perf.CM],'Cortical',textColorThresh);
    
    % simulation info
    annotation('textbox',[.75 .85 .2 .1],...
           'string',data(z).annot(vv,3:end),...
           'FitBoxToText','on',...
           'LineStyle','none')

    % save grid
    Dirparts = strsplit(study_dir, filesep);
    DirPart = fullfile(Dirparts{1:end-1});
    saveas(gca,[DirPart filesep 'SpatialGrid vary ' variedParam num2str(varies(end).range(vv),'%0.2f') '.tiff'])
    clf
end
set(0, 'DefaultFigureVisible', 'on')
