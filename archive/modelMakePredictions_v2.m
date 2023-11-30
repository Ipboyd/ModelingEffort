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
expName = '001 vary TD-E gsyn';

% setup directory for current simulation
datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);
if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));
simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

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

varies(2) = []; % remove redundant vary

varies(end+1).conxn = 'TD->X';
varies(end).param = 'gSYN';
varies(end).range = 0.05;

varies(end+1).conxn = 'TD->R';
varies(end).param = 'gSYN';
varies(end).range = 0.03;

varies(end+1).conxn = 'Exc->R';
varies(end).param = 'gSYN';
varies(end).range = 0.23;

varies(end+1).conxn = 'TD';
varies(end).param = 'Itonic';
varies(end).range = 1.5:0.01:1.55;

% display parameters
[{varies.conxn}' {varies.param}' {varies.range}']

varied_param = find(cellfun(@length,{varies.range})>1);
if length(varied_param) > 1, varied_param = varied_param(2); end
expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');

% specify netcons
netcons.xrNetcon = ones(4)-eye(4); % cross channel inhibition
netcons.irNetcon = eye(4); %inh -> R
netcons.tdxNetcon = eye(4); % I2 -> I
netcons.tdrNetcon = eye(4); % I2 -> R

%%%%%%%%%%%%%%%%%%%%%%%%%%%% the slow way %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% varies(end+1).conxn = '(Inh->Inh,Exc->Exc)';
% varies(end).param = 'locNum';
% varies(end).range = subz;
% 
% %run network
% options.time_end = 2700;
% optinos.parfor_flag = 1;
% [temp,s] = mouseNetwork(study_dir,varies,netcons,options);
%     
%     % post process
%     [data(z).perf,data(z).fr] = postProcessData(temp,options);


%%%%%%%%%%%%%%%%%% the fast way: concatenate over time %%%%%%%%%%%%%%%%%



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
options.locNum = [];
options.parfor_flag = 1;
[temp,s] = mouseNetwork(study_dir,varies,netcons,options);

% TD firing rates
FR_TD = zeros(1,length(varies(varied_param).range));
for i = 1:length(varies(varied_param).range)
    FR_TD(i) = max(sum(temp(i).TD_V_spikes)/options.time_end*1000);
    sprintf('%.02f: %03f',temp(i).TD_Itonic,FR_TD(i))
end
figure;plot(varies(varied_param).range,FR_TD)
xlabel(expVar)
ylabel('FR, Hz')

% post-process, calculate performance
data = struct();
dataOld = struct();
options.time_end = 3000;
PPtrialStartTimes = [1 cumsum(trialStartTimes)+1];
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime-options.time_end+1);
options.plotRasters = 0;
options.subPops = {'X','R','C'}; %individually specify population performances to plot
configName = cellfun(@(x) strsplit(x,'_'),{ICfiles(subz).name}','UniformOutput',false);
configName = vertcat(configName{:});
configName = configName(:,1);
options.variedField = strrep(expVar,'-','_');
tic
parfor z = 1:length(subz)
    trialStart = PPtrialStartTimes(z);
    trialEnd = PPtrialEndTimes(z);
    figName = [simDataDir filesep configName{z}];
    [data(z).perf,data(z).fr] = postProcessData_new(temp,s,trialStart,trialEnd,figName,options);
    
%     [dataOld(z).perf,dataOld(z).fr] = postProcessData(temp,trialStart,trialEnd,options);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
vizNetwork(s,0,'C','Exc')

% data = dataOld;
% plotPerformanceGrids;

% options.subPops = {'C'};
plotPerformanceGrids_new;
