% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

% note:
% large bottleneck lies in r/w to network drive

if ~contains(pwd,'ICSimStim'), cd('ICSimStim'); end
clearvars
clc
close all

addpath(genpath('strflab_v1.45'))
addpath('../genlib')
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
strfGain = 0.1; % 0.1;

% frequency parameters - static

% % broadband
paramG.BW = 2000; %2000;  % Hz
paramG.f0 = 4300; %4300; % ~strf.f(30)
paramG.BSM = 5.00E-05; % 1/Hz=s best spectral modulation

%% load vocoded speech envelopes

load('Penikis2023_vocodedspeech_amplitudeenvelopes.mat');

fs_stim = 20000;

nStim = size(VS_stim_full,1);

t_zero = 0.250; % zero-pad front by 250 ms so that STRF convolution doesn't cut out stim
VS_sig = zeros(1,round(t_zero*fs_stim));
full_env = zeros(1,round(t_zero*fs_stim));

for n = 1:nStim

    % zero-pad remaining envelpe

    env = VS_stim_full(n,:);

    % upsample envelope to 20000 Hz
    env = interp(env,fs_stim/fs); env(isnan(env)) = 0;
    nonzero = find(env);
    
    % truncate zeroes after token
    env = env(1:nonzero(end));

    t_stim = length(env)/fs_stim;

    % add inter-stimulus interval of 0.5 seconds after vocoded speech
    % snippet
    t_after = 0.5;
    env(end+1:round(t_stim*fs_stim)) = 0;

    carrier = randn(1,round(t_stim*fs_stim));

    raw_sig = env .* bandpass(carrier,[100 9000],fs_stim);
    raw_sig = [raw_sig , zeros(1,round(t_after*fs_stim))];

    VS_sig = cat(2,VS_sig,targetlvl*raw_sig/rms(raw_sig));

    % store full_env for peakDrv calculations
    full_env = cat(2,full_env,[env,zeros(1,round(t_after*fs_stim))]);

end

[~,env,peakDrv_vec,~] = find_peakRate(full_env,fs_stim,'amp_linear');

% remove all peakDrv events that follow silence (where env == -Inf)
inds = find(peakDrv_vec);
inds(env(inds-1) == -Inf) = [];

% calculate peakDrv events in full stimulus (in ms)
peakDrv = inds / fs_stim * 1000;

[VS_spec,t,f] = STRFspectrogram(VS_sig,fs_stim);
specs.songs = VS_spec;

specs.dims = size(VS_spec);
specs.t = t;
specs.f = f;
t_vec_stim = (0:length(VS_sig)-1)/fs_stim;

save('Penikis_VocodedSpeech.mat',"VS_sig",'fs_stim','t_vec_stim','t_stim','peakDrv');

strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));

strfGain_new = strfGain * sum(strf.w1,'all') / 43.2;
trf.w1 = strfGain_new * strf.w1/strfGain;

paramSpk.t_ref = 1.5;
paramSpk.t_ref_rel = 0.5;
paramSpk.rec = 4;

%% Run simulation script
mean_rate = 0.1;

tuningParam.strf = strf;
tuningParam.type = tuning;
tuningParam.sigma = sigma;

[fr_target_on,fr_target_off] = STRFconvolve_V2(strf,specs.songs*stimGain,mean_rate);

save('VocodedSpeech_FR_traces.mat','fr_target_on','fr_target_off','paramH','paramG','strfGain');
