%% Initialize

mfileinfo = mfilename('fullpath');
mfiledir = fileparts(mfileinfo);
cd(mfiledir);

dynasimPath = '../DynaSim';

addpath('mechs'); addpath('resampled-stimuli'); addpath(genpath('ICSimStim'));
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');

load('default_STRF_with_offset_200k.mat');

dt = 0.1; %ms

% study_dir: folder under 'run' where m files and input spikes for simulations are written and saved
study_dir = fullfile(pwd,'run','single-channel-offset');

if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));

% expName: folder under 'simData' where results are saved
expName = '07-31-23';

% for newStrfGain = strfGains
% simDataDir = [pwd filesep 'simData' filesep expName ' ' num2str(newStrfGain)];

newStrfGain = strfGain;
simDataDir = [pwd filesep 'simData' filesep expName];

if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

%% Run .m file to generate options and varies structs for simulations
addpath('params');
% addpath('params_noiseonly');

% params_MaskedPerf;
% params_Masked_varyOnsetPV;
% params_Masked_varyOnsetNoise;
% params_DepressiveStr;
params_ExpFig3;

% for figures in paper

%Figure 4a, params_4a;
%Figure 4b, params_4b;
% Figure 4c, params_4c;

% For Figure 5, params_5
% For Figure 6, params_6
% For Figure 7, params_7
% for Figure 8, params_8

%% create input spikes from STRFs
% concatenate spike-time matrices, save to study dir

prepInputData;

%% run simulation

options.strfGain = newStrfGain;

if isempty(options.locNum), options.time_end = size(spks,1)*dt; %ms;
else, options.time_end = padToTime; end
[snn_out,s] = columnNetwork_V2(study_dir,varies,options,netcons);

%% post-process for performance and firing results

postProcessSims;

% end
