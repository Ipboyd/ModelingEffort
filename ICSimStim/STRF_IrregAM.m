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

strfGain = 0.08; % 0.1;

% frequency parameters - static
paramG.BW = 3000; %2000;  % Hz
paramG.BSM = 5.00E-05; % 1/Hz=s best spectral modulation
paramG.f0 = 5500; %4300; % ~strf.f(30)

%% generate AM stimuli and calc spectrograms

fs = 20000;    % sampling rate of stimuli should be equal to 2*spectrogram sampling rate (10000Hz)
%load('irregular_stim.mat');
load('bandlimited_stim.mat')

for n = 1:2
    % zero pad AM sigs by 250ms
    [specs.songs{n},t,f] = STRFspectrogram(AM_sig{n}/rms(AM_sig{n})*targetlvl,fs);
end
specs.dims = size(specs.songs{1});
specs.t = t;
specs.f = f;

% make STRF
strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));
strf.w1 = strf.w1*strfGain;

paramSpk.t_ref = 1.5;
paramSpk.t_ref_rel = 0.5;
paramSpk.rec = 4;

%% Run simulation script
mean_rate = 0.1;

tuningParam.strf = strf;
tuningParam.type = tuning;
tuningParam.sigma = sigma;

for a = 1:length(specs.songs)
    [~,~,fr_target_on{a},fr_target_off{a}] = STRFconvolve_V2(strf,specs.songs{a}*stimGain,mean_rate,1,[],paramSpk.t_ref,paramSpk.t_ref_rel,paramSpk.rec);
end

save('bandlimited_AM_FR_traces.mat','fr_target_on','fr_target_off','paramH','paramG','strfGain');