% re-create grids fitted by Jio. 
% Modulate with top-down signals & make predictions.
%
% assuming STRF response is available in ICresponseDir
% restructure STRF response from spike times to spike trains

% Kenny Chou
% 2020-11-17
%% 1 - re-create fitted grids

% paths
dynasimPath = 'C:\Users\Kenny\Desktop\GitHub\DynaSim';
fittedDataPath = 'data_fitted\20201113'; %path to parameters Jio used
ICresponseDir = 'data_fitted\s30_STRFgain14.42_2020'; %path to STRF respones Jio used
ICdir = [pwd filesep ICresponseDir filesep 'spikeTrains'];
addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))

% check IC inputs
if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

% load network parameters
load([fittedDataPath filesep 'network_params.mat'],'varies','netcons','restricts')
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_2020');
options.plotRasters = 0;
% netcons.xrNetcon = zeros(4);

% get indices of STRFS, all locations, excitatory inputs only
ICfiles = dir([ICdir filesep '*.mat']);
% subz = find(contains({ICfiles.name},'m0') & contains({ICfiles.name},'_E')); % target only
subz = find(~contains({ICfiles.name},'s0') & contains({ICfiles.name},'_E')); % all

datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);

data = struct();
for z = 1:length(subz)
    % run network
%     ic1 = load([ICdir filesep ICfiles(subz(z)).name]);
%     ic2 = load([ICdir filesep ICfiles(subz(z)+1).name]);
%     options.time_end = min([size(ic1.spks,3),size(ic2.spks,3)]); %ms
    options.time_end = 2700;
    options.locNum = subz(z);
    [temp,s] = mouseNetwork(study_dir,varies,netcons,restricts,options);
    
    % post process
    [data(z).perf,data(z).fr] = postProcessData(temp,options);
end

% temp = data;
% for z = 1:length(subz)
%     data.perf(z).C = temp(z).perf.C.C;
%     data.fr(z).C = temp(z).fr.C.C;
% end
%% performance grids
% performance vector has dimensions [numSpatialChan,nvaried]
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

fileNames = {ICfiles.name};
targetIdx = find(contains(fileNames,'m0') & contains({ICfiles.name},'_E')); %target only
maskerIdx = find(contains(fileNames,'s0') & contains({ICfiles.name},'_E')); %masker only
mixedIdx = find(~contains(fileNames,'m0') & ~contains(fileNames,'s0') & contains({ICfiles.name},'_E'));

h = figure('position',[200 200 600 600]);

clear perf fr
vv = 1;

% C neuron; mixed cases
if sum(ismember(subz,mixedIdx)) > 0
    for i = 1:length(mixedIdx)
        idx = (mixedIdx(i) == subz);
        perf.C(i) = data(idx).perf.C(vv);
        fr.C(i) = data(idx).fr.C(vv);
    end
    subplot('Position',[0.4 0.15 0.45 0.35])
    plotPerfGrid(perf.C',fr.C',[]);
end

% C neuron; target or masker only cases
if sum(ismember(subz,targetIdx)) > 0
    perf.CT = zeros(1,4);
    perf.CM = zeros(1,4);
    fr.CT = zeros(1,4);
    fr.CM = zeros(1,4);
    if ~isempty(targetIdx)
        for i = 1:length(targetIdx)
            idx = (targetIdx(i) == subz);
            perf.CT(i) = data(idx).perf.C(vv);
            fr.CT(i) = data(idx).fr.C(vv);
        end
    end    
    if ~isempty(maskerIdx)
        for i = 1:length(maskerIdx)
            idx = (maskerIdx(i) == subz);
            perf.CM(i) = data(i).perf.C(vv);
            fr.CM(i) = data(i).fr.C(vv);
        end
    end    
    subplot('Position',[0.4 0.6 0.45 0.2])
    plotPerfGrid([perf.CT;perf.CM],[fr.CT;fr.CM],'Cortical');
end

% % simulation info
% annotation('textbox',[.8 .85 .15 .2],...
%        'string',data(z).annot(vv,3:end),...
%        'FitBoxToText','on',...
%        'LineStyle','none')

% save grid
% Dirparts = strsplit(study_dir, filesep);
% DirPart = fullfile(Dirparts{1:end-1});
% saveas(gca,[DirPart filesep 'SpatialGrid vary ' variedParam num2str(varies(end).range(vv),'%0.2f') '.tiff'])
% clf

