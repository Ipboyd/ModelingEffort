%% Initialize

% mfileinfo = mfilename('fullpath');
% mfiledir = fileparts(mfileinfo);
% cd(mfiledir);

dynasimPath = ['..' filesep 'DynaSim'];

addpath('mechs'); addpath(genpath('ICSimStim'));
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');
addpath('params-AM');

dt = 0.1; %ms

% study_dir: folder under 'run' where m files and input spikes for simulations are written and saved
study_dir = fullfile(pwd,'run','single-channel-AM-stim');

if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));

load('AM_stim_FR_traces.mat');
load('peakDrv_AM.mat');
load('AM_sigs.mat')

% expName: folder under 'simData' where results are saved
expName = '08-30-23 fix dynamics, vary strength';

% for newStrfGain = strfGains
% simDataDir = [pwd filesep 'simData' filesep expName ' ' num2str(newStrfGain)];

strfGain = 0.08; %strfGain;
simDataDir = [pwd filesep 'simData-AM' filesep expName];

if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

padToTime = (t_stim + 0.250 + 0.250) * 1000; %ms, first 250ms accts for padding in STRF_AMStim, second 250ms is for padding at the end

if length(fr_target_on) > 2 % AM stim with steady modulation
    nStim = 5;
    nTrials = 100;
    TrialsPerStim = nTrials/5;
else
    nStim = 2; % irregular AM stim
    nTrials = 20;
    TrialsPerStim = nTrials/2;
end

%% Run .m file to generate options and varies structs for simulations

% params_AM_adjustPVDynamics;
% params_AM_noDep;
% params_AM_noPV;
params_AM_varyingStrengths;


%% create input spikes from STRFs
% concatenate spike-time matrices, save to study dir

options.regenSpks = 0;
prepInputData_AMStim;

%% run simulation

if isempty(options.locNum), options.time_end = size(spks,1)*dt; %ms;
else, options.time_end = padToTime; end
[snn_out,s] = columnNetwork_V2(study_dir,varies,options,netcons);

%% post-process for performance and firing results

options.strfGain = strfGain; % store in options for logging
postProcessAM;
