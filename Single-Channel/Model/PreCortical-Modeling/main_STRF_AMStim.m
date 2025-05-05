% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

% note:
% large bottleneck lies in r/w to network drive
tic
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
%paramH.N2 nominal = 7
paramH.N2 = 8;
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

%% generate AM stimuli and calc spectrograms

AM_freqs = [2 4 8 16 32]; % divide by 2 since phase and antiphase are the same in envelope
fs = 20000;    % sampling rate of stimuli should be equal to 2*spectrogram sampling rate (10000Hz)
t_stim = 3; % at least 5 full cycles per trial for 2 hz modulation
t_zero = 0.250; % zero-pad by 250 ms so that STRf convolution doesn't cut out stim

% create and zero-pad carrier signal
carrier = randn(round(t_stim*fs),1); t_vec = (0:(length(carrier)-1))/fs;
carrier = [zeros(round(t_zero*fs),1); bandpass(carrier,[100 9000],fs)];

% 1st row of AM_sigs is unmodulated white noise
AM_sig = carrier;

AM_spec = STRFspectrogram(AM_sig/rms(AM_sig)*targetlvl,fs);
specs.songs{1} = AM_spec;
AM_sigs(1,:) = AM_sig;

for a = 1:length(AM_freqs)
    envelope = [ zeros(round(t_zero*fs),1); -cos(2*pi*AM_freqs(a)*t_vec')+1];

    % zero pad first 250ms of AM stim to account for first 250ms of nans in
    % STRF calculations
    AM_sig = carrier.*envelope;

    [AM_spec,t,f] = STRFspectrogram(AM_sig/rms(AM_sig)*targetlvl,fs);
    specs.songs{a+1} = AM_spec;
    AM_sigs(a+1,:) = AM_sig;
    
    % get envelope using hilbert, convert to db
    dbsig = 20*log10(abs(envelope));

    t_vec_db = (0:(length(envelope)-1))/fs;

    % find local minima and exclude all minima at the zero-padded 250 ms
    mins = islocalmin(dbsig); mins(1:round(fs*t_zero)) = 0;

    % find peakDrv events (in ms)
    peakDrv{a} = round(1000*t_vec_db(mins));
end

specs.dims = size(AM_spec);
specs.t = t;
specs.f = f;
t_vec_stim = (0:size(AM_sigs,2)-1)/fs;
save('preprocessed_AM_V2.mat','specs')
save('AM_sigs.mat',"AM_sigs",'fs','t_vec_stim','t_stim');
save('peakDrv_AM.mat','peakDrv');

% load('preprocessed_AM_V2.mat')

% make STRF

% f0s = [500 1000 2000 4000];

% for f = 1:length(f0s)

% paramG.f0 = f0s(f);
% bandwidth is 0.5*octave
% paramG.BW = (sqrt(2)*f0s(f) - f0s(f)/sqrt(2)) / (2*sqrt(2*log(2))); % estimate FWHM and convert back to standard deviation

strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));
% strf.w1 = strf.w1*strfGain;

strfGain_new = strfGain * sum(strf.w1,'all') / 43.2;
trf.w1 = strfGain_new * strf.w1/strfGain;

% plot(1000*strf.t,sum(strf.w1)); hold on;
% xlabel('Time (ms)'); ylabel('Weight'); title('MGB STRF (Figure 3c)')
% savefig(gcf,'MGB STRF Figure 3c, v3');

% sum of STRF with gain should be ~43.2;
% adjust STRF gain for spiking

paramSpk.t_ref = 1.5;
paramSpk.t_ref_rel = 0.5;
paramSpk.rec = 4;

%% Run simulation script
mean_rate = 0.1;

tuningParam.strf = strf;
tuningParam.type = tuning;
tuningParam.sigma = sigma;

for a = 1:length(specs.songs)
    [fr_target_on{a},fr_target_off{a}] = STRFconvolve_V2(strf,specs.songs{a}*stimGain,mean_rate);
end

% save(sprintf('AM_stim_FR_traces_cf%dHz.mat',f0s(f)),'fr_target_on','fr_target_off','paramH','paramG','strfGain');
save('AM_stim_FR_traces_narrower.mat','fr_target_on','fr_target_off','paramH','paramG','strfGain');
% end
toc