% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('../genlib')

% Spatial tuning curve parameters
sigma = 30; %60 for bird but 38 for mouse
tuning = 'mouse';

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
datetime=['v2' filesep datestr(now,'HHMMSS') '_seed142307_s' num2str(sigma)];
stimGain = 1;
saveFlag = 0;

songLocs = 1:4;
maskerLocs = 1:4;

saveParam.flag = 1;
saveParam.fileLoc = datetime;
tuningParam.type = tuning;
tuningParam.sigma = sigma;
tuningParam.H = paramH;
tuningParam.G = paramG;

for songloc = songLocs
    close all
    maskerloc=0;
    t_spiketimes=InputGaussianSTRF_v2(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain);
    t_spiketimes=InputGaussianSTRF_v2(maskerloc,songloc,tuningParam,saveParam,mean_rate,stimGain);
    for maskerloc = maskerLocs
        t_spiketimes=InputGaussianSTRF_v2(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain);
    end
end

%% Grids for each neuron
fileloc = [tuning filesep saveParam.fileLoc];
allfiles = cellstr(ls([fileloc filesep '*.mat']))
tgtalone = cellstr(ls([fileloc filesep '*m0.mat']))
mskalone = cellstr(ls([fileloc filesep 's0*.mat']))
mixedfiles = setdiff(allfiles,[tgtalone;mskalone])
for i = 1:16
    data = load([fileloc filesep mixedfiles{i}]);
    perf(i,:) = data.disc;
end

neurons = {'left sigmoid','gaussian','u','right sigmoid'};
[X,Y] = meshgrid(songLocs,fliplr(maskerLocs));
for i = 1:length(neurons)
    neuronPerf = perf(:,i);
    str = cellstr(num2str(round(neuronPerf)));
    neuronPerf = reshape(neuronPerf,4,4);
    figure;
    imagesc(flipud(neuronPerf));
    xticks([1:4]); xticklabels({'-90','0','45','90'})
    yticks([1:4]); yticklabels(fliplr({'-90','0','45','90'}))
    title(neurons(i))
    text(X(:)-0.2,Y(:),str)
    caxis([50,100])
    xlabel('Song Location')
    ylabel('Masker Location')
    saveas(gca,[fileloc filesep neurons{i} '.tiff'])
end
