% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

% note:
% large bottleneck lies in r/w to network drive

cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim');
clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('../genlib')
addpath('../fixed-stimuli')
addpath('../plotting')
addpath('../cSPIKE')
InitializecSPIKE;
dataSaveLoc = pwd; %local save location

% Spatial tuning curve parameters
sigma = 24; %60 for bird but 38 for mouse
tuning = 'mouse'; %'bird' or 'mouse'
stimGain = 0.5;
targetlvl = 0.01;
maskerlvl = 0.01; %default is 0.01
maxWeight = 1; %maximum mixed tuning weight; capped at this level.

% load stimuli & calc spectrograms

% [song1,~] = audioread('200k_target1.wav');
% [song2,~] = audioread('200k_target2.wav');

% for trial = 1:10
%     [masker,fs] = audioread(['200k_masker' num2str(trial) '.wav']);
%     [spec,~,~] = STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
%     masker_specs{trial} = spec;
% end

% alpha_var = 0.008 : 0.0004 : 0.012;
% N1_var = 1 : 7
% N2_var = N1_var + 2;
% SC2_var = 1 : -0.02 : 0.8

% gain_var = 0.09 : 0.005 : 0.13;

% % temporal parameters - all free but SC1
% paramH.alpha = 0.0105; % time constant of temporal kernel [s] 0.0097
% paramH.N1 = 2; %5
% paramH.N2 = 5; %7
% paramH.SC1 = 1;
% paramH.SC2 = 0.7;  % increase -> more inhibition 0.88
% strfGain = 0.05; % 0.10

paramH.alpha = 0.01; % time constant of temporal kernel [s] 0.0097
paramH.N1 = 5;
paramH.N2 = 7;
paramH.SC1 = 1;
paramH.SC2 = 0.88;  % increase -> more inhibition

% % wider MGB
% paramH.alpha = 0.0105; % time constant of temporal kernel [s] 0.0097
% paramH.N1 = 4; %5
% paramH.N2 = 7; %7
% paramH.SC1 = 1;
% paramH.SC2 = 0.85;  % increase -> more inhibition 0.88

strfGain = 0.1;

% frequency parameters - static
paramG.BW = 2000;  % Hz
paramG.BSM = 5.00E-05; % 1/Hz=s best spectral modulation
paramG.f0 = 4300; % ~strf.f(30)

load('preprocessed_stims.mat'); % don't need to run STRFspectrogram again
% [song1_spec,t,f]=STRFspectrogram(song1/rms(song1)*targetlvl,fs);
% [song2_spec,~,~]=STRFspectrogram(song2/rms(song2)*targetlvl,fs);
% specs.songs{1} = song1_spec;
% specs.songs{2} = song2_spec;
% specs.maskers = masker_specs;
% specs.dims = size(song1_spec);
% specs.t = t;
% specs.f = f;

% make STRF
strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));
strf.w1 = strf.w1*strfGain;

plot(1000*strf.t,sum(strf.w1)); hold on;
xlabel('Time (ms)'); ylabel('Weight'); title('MGB STRF (Figure 3c)')
% savefig(gcf,'MGB STRF Figure 3c, v3');

% sum of STRF with gain should be ~43.2;
% adjust STRF gain for spiking

%strfGain_new = strfGain * sum(strf.w1,'all') / 43.2;
%strf.w1 = strfGain_new * strf.w1/strfGain;

paramSpk.t_ref = 1.5;
paramSpk.t_ref_rel = 0.5;
paramSpk.rec = 4;

%% Run simulation script
mean_rate = 0.1;

tuningParam.strf = strf;
tuningParam.type = tuning;
tuningParam.sigma = sigma;

[~,~,fr_target_on{1},fr_target_off{1}] = STRFconvolve_V2(strf,specs.songs{1}*stimGain,mean_rate,1,[],paramSpk.t_ref,paramSpk.t_ref_rel,paramSpk.rec);
[~,~,fr_target_on{2},fr_target_off{2}] = STRFconvolve_V2(strf,specs.songs{2}*stimGain,mean_rate,1,[],paramSpk.t_ref,paramSpk.t_ref_rel,paramSpk.rec);

for m = 1:10
    [~,~,fr_masker{m}] = STRFconvolve_V2(strf,specs.maskers{m}*stimGain,mean_rate,1,[],paramSpk.t_ref,paramSpk.t_ref_rel,paramSpk.rec);
end

save('default_STRF_with_offset.mat','fr_target_on','fr_target_off','fr_masker','paramH','paramG','strfGain');