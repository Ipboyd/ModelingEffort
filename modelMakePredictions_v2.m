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

% setup directory for current simulation
datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);
if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));

% get indices of STRFS, all locations, excitatory inputs only
ICfiles = dir([ICdir filesep '*.mat']);
% subz = find(contains({ICfiles.name},'m0') & contains({ICfiles.name},'_E')); % target only
subz = find(~contains({ICfiles.name},'s0') & contains({ICfiles.name},'_E')); % all

% check IC inputs
if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

% load network parameters
load([fittedDataPath filesep 'network_params.mat'],'varies','netcons')
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_2020');
options.plotRasters = 0;

% custom parameters
varies(1).conxn = '(Inh->Inh,Exc->Exc)';
varies(1).param = 'trial';
varies(1).range = 1:20;

varies(2) = [];

varies(end+1).conxn = 'TD';
varies(end).param = 'Itonic';
varies(end).range = 7;

% specify netcons
netcons.xrNetcon = ones(4)-eye(4); % cross channel inhibition
netcons.irNetcon = eye(4); %inh -> R
netcons.tdxNetcon = eye(4); % I2 -> I
netcons.tdrNetcon = zeros(4); % I2 -> R

%%%%%%%%%%%%%%%%%%%%%% the slow way %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data = struct();
% for z = 1:length(subz)
%     % run network
%     options.time_end = 2700;
%     options.locNum = subz(z);
%     [temp,s] = mouseNetwork(study_dir,varies,netcons,options);
%     
%     % post process
%     [data(z).perf,data(z).fr] = postProcessData(temp,options);
% end


%%%%%%%%%%%%%%%%%%%%%%%%% the fast way %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% concatenate spike-time matrices, save to study dir
trialDur = zeros(1,length(subz));
trialStartTimes = zeros(1,length(subz));
padToTime = 3300; %ms
label = {'E','I'};
for ICtype = 0:1
    spks = [];
    for z = 1:length(subz)
        disp(ICfiles(subz(z)+ICtype).name)
        spkTrain = load([ICdir filesep ICfiles(subz(z)+ICtype).name]);
        trialDur(z) = size(spkTrain.spks,3);
        trialStartTimes(z) = padToTime;
        % pad each trial to have duration of timePerTrial
        if size(spkTrain.spks,3) < padToTime
            padSize = padToTime-size(spkTrain.spks,3);
            spkTrain.spks = cat(3,spkTrain.spks,zeros(20,4,padSize)); 
        end
        % concatenate
        spks = cat(3,spks,spkTrain.spks);
    end
    save(fullfile(study_dir, 'solve',sprintf('IC_spks_%s.mat',label{ICtype+1})),'spks');
end

% run simulation
options.time_end = size(spks,3);
[temp,s] = mouseNetwork(study_dir,varies,netcons,options);
FR_TD = median(sum(temp(1).TD_V_spikes)/options.time_end*1000);

data = struct();
options.trialStartTimes = [1 cumsum(trialStartTimes)+1];
options.plotRasters = 0;
options.time_end = 3000;
subPops = {'R','C'}; %individually specify population performances to plot
for z = 1:length(subz)
    % post process
    options.trialStart = options.trialStartTimes(z);
    options.trialEnd = options.trialStartTimes(z+1)-(padToTime-options.time_end+1);
    options.subPops = subPops;
    [data(z).perf,data(z).fr] = postProcessData(temp,options);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
vizNetwork(s,0,'C','Exc')

plotPerformanceGrids;
