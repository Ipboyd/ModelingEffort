% This mainscript will help you create Figures 3 through 5 in the bioRxiv
% manuscript. - Jio
tic;
plot_all = 1;

%% Initialize

% change current directory to folder where this script is stored
mfileinfo = mfilename('fullpath');
mfiledir = strsplit(mfileinfo,filesep);
% cd(fullfile(mfiledir{1:end-1}));

%dynasimPath = fullfile('..','DynaSim');
dynasimPath = 'DynaSim-master';


addpath('mechs');
addpath('resampled-stimuli');
addpath(genpath('ICSimStim'));
addpath('genlib');
addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');
addpath('subfunctions');

%% Make ICfiles.mat if it's not in your directory

if ~isfile('ICfiles.mat'), makeICfiles; end

%% user inputs
dt = 0.1; %ms, should be a multiple of 0.1 ms

% study_dir: folder under 'run' where m files and input spikes for simulations are written and saved
study_dir = fullfile(pwd,'run','1-channel-paper');

if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
%mkdir(fullfile(study_dir, 'solve'));


solve_directory = fullfile(study_dir, 'solve');
% 
if exist(fullfile(study_dir, 'solve'), 'dir')
    %don't remove the directory
    flag_raised_mex = 1;
else
    if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
    mkdir(solve_directory); 
    flag_raised_mex  = 0;

    mexes_dir = fullfile(mfiledir{1:end-1}, 'mexes');
    if isfolder(mexes_dir)
        %These might need to be changed for this model!!!!
        m_file_to_copy = 'solve_ode_1_channel_paper.m';
        mex_file_to_copy = 'solve_ode_1_channel_paper_mex.mexw64';
        mex_file_path = fullfile(mexes_dir, mex_file_to_copy);
        mex_files = dir([mex_file_path, '.*']);
        if ~isempty(mex_files)
            flag_raised_mex = 1;

            end
        end
end

addpath(solve_directory);


% expName: folder under 'simData' where results are saved
expName = '12-18-23 test run';
simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

%% Run .m file to generate options and varies structs for simulations
addpath('params');
%addpath('params-AM')
%params_5_rate_based_onoff_WorkingCopy; % Generates Figure 5 (rate-based simulation)
%params_5_rate_based_onoff_WorkingCopy_HalfDense;

%params_4_opto_onoff_fig6;
%params_5_rate_based_onoff_offNonSupressed;
%params_opto_onoff_2
%params_5_off_dominated
%onoff_con;
%params_AM_best_onoff;

params_5_rate_based_onoff_fig4;
%params_off;
%params_both;
%params_on_Laser;
%params_no_pv;



%% create spatially-tuned channels based on options.nCells

% spatial tuning at inputs
[azi,spatialCurves,chanLabels,bestLocs] = genSpatiallyTunedChans(options.nCells);

% use a separate struct for connectivity matrices (netcons) between populations
% row = source, column = target
netcons = struct; 

% PEnetcon: PV->E, model as Gaussians for now
sigma = 30;
netcons.PEnetcon = makePENetcon(bestLocs,sigma);
netcons.XRnetcon = eye(1);
netcons.RCnetcon = eye(1);

%% load input stimuli (targets and maskers) from ICSimStim
load('default_STRF_with_offset_200k.mat');

% firing rates were generated with sampling rate of 10000 Hz to match old
% simulation time step, downsample if dt's don't match
if dt ~= 0.1
    dsamp_fac = dt/0.1;
    for m = 1:10    
        fr_masker{m} = downsample(fr_masker{m},dsamp_fac);
    end
    for t = 1:2
        fr_target_on{t} = downsample(fr_target_on{t},dsamp_fac);
        fr_target_off{t} = downsample(fr_target_off{t},dsamp_fac);
    end
end

% edit strfGain if you want to rescale firing rate at inputs
newStrfGain = strfGain;

%% create input spikes from STRFs

padToTime = 3500; % [ms]

% ICfiles.mat contains names of spatial grid configs: s[targetloc]m[maskerloc]
% See 'config_idx_reference.JPG' for indexes
% options.locNum is defined in params .m file
% if it's empty, default to running all 24 configs (including masker-only
% trials)

if ~isempty(options.locNum)
    subz = options.locNum;
else
    subz = 1:24;
end

% concatenate spike-time matrices, save to study_dir
prepInputData;

%% run simulation

options.strfGain = newStrfGain;
options.dt = dt;

if isempty(options.locNum), options.time_end = size(spks,1)*dt; % [ms];
else, options.time_end = padToTime*numel(options.locNum); end
%[snn_out,s] = columnNetwork_paper_onoff_Excitatory(study_dir,varies,options,netcons,flag_raised_mex);

%Going to try rerunning old stuff to see if it is broken.
[snn_out,s] = columnNetwork_paper_onoff(study_dir,varies,options,netcons,flag_raised_mex);
%[snn_out,s] = columnNetwork_paper_on_only_nopv(study_dir,varies,options,netcons,flag_raised_mex);
%[snn_out,s] = columnNetwork_paper_onoff_off_Conv(study_dir,varies,options,netcons,flag_raised_mex);
%[snn_out,s] = columnNetwork_paper_onoff_Both_Conv(study_dir,varies,options,netcons,flag_raised_mex);



%% post-process for performance and firing results

postProcessSims;
toc;

%Monitors
FR = data(15).fr.R2On.channel1;
perf = data(15).perf.R2On.channel1;


% figure;
% subplot(2,1,1)
% plot(snn_out(1).R1On_V(1:35000))
% subplot(2,1,2)
% plot(snn_out(1).S1OnOff_V(1:35000))