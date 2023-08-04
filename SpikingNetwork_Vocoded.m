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

study_dir = fullfile(pwd,'run','single-channel-vocoded-speech');

if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));

load('Penikis_VocodedSpeech.mat');
load('VocodedSpeech_FR_traces.mat');

% expName: folder under 'simData' where results are saved
expName = '08-03-23 vocoded speech response, no PV inhibition';

strfGain = 0.08; %strfGain;
simDataDir = [pwd filesep 'simData-Vocoded' filesep expName];

if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

t_stim = length(VS_sig)/fs_stim;
padToTime = (t_stim + 0.250) * 1000; %ms, first 250ms accts for padding in STRF_AMStim

nTrials = 20;

%% Run .m file to generate options and varies structs for simulations

% params_AM_adjustPVDynamics;
params_AM_noPV;

%% create input spikes from STRFs
% concatenate spike-time matrices, save to study dir

options.locNum = [];
options.regenSpks = 0;
prepInputData_Vocoded;

%% run simulation

if isempty(options.locNum), options.time_end = size(spks,1)*dt; %ms;
else, options.time_end = padToTime; end
[snn_out,s] = columnNetwork_V2(study_dir,varies,options,netcons);

%% post-process for performance and firing results

options.strfGain = strfGain; % store in options for logging
postProcessVocoded;
