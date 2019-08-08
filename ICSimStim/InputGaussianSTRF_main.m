% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('../genlib')

% Spatial tuning curve parameters
sigma = 20; %60 for bird but 38 for mouse
tuning = 'bird';

% STRF parameters
paramH.t0=7/1000; % s
paramH.BW=0.0045; % s temporal bandwith (sigma: exp width)
paramH.BTM=56;  % Hz  temporal modulation (sine width)
paramH.phase=.49*pi;

paramG.BW=2000;  % Hz
paramG.BSM=5.00E-05; % 1/Hz=s
paramG.f0=4300;

%% Run simulation script
mean_rate=.1;
datetime=datestr(now,'HHMMSS');
stimGain = 0.4;
saveFlag = 0;

songLocs = 3;
maskerLocs = 3;

saveParam.flag = 0;
saveParam.fileLoc = datetime;
tuningParam.type = tuning;
tuningParam.sigma = sigma;
tuningParam.H = paramH;
tuningParam.G = paramG;

for songloc = songLocs
    close all
%     maskerloc=0;
%     t_spiketimes=InputGaussianSTRF(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain);
%     t_spiketimes=InputGaussianSTRF(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain);
    for maskerloc = maskerLocs
        t_spiketimes=InputGaussianSTRF(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain);
    end
end