% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('../genlib')
dataloc = 'Z:\eng_research_hrc_binauralhearinglab\kfchou\ActiveProjects\MiceSpatialGrids\ICStim';
% Spatial tuning curve parameters
sigma = 30; %60 for bird but 38 for mouse
tuning = 'mouse';
stimGain = 0.5;
maskerlvl = 0.01; %default is 0.01

% STRF parameters - don't need to change
paramH.t0=7/1000; % s
paramH.BW=0.0045; % s temporal bandwith (sigma: exp width)
paramH.BTM=56;  % Hz  temporal modulation (sine width)
paramH.phase=.49*pi;

paramG.BW=2000;  % Hz
paramG.BSM=5.00E-05; % 1/Hz=s
paramG.f0=4300;

%% Run simulation script
for maskerlvl = 0.001%:0.002:0.03
mean_rate=.1;
saveName=['s' num2str(sigma) '_sg' num2str(stimGain) '_ml' num2str(maskerlvl) '_' datestr(now,'YYYYmmdd-HHMMSS')];
saveFlag = 0;

songLocs = 1:4;
maskerLocs = 1:4;

saveParam.flag = 1;
saveParam.fileLoc = [dataloc filesep tuning filesep saveName];
tuningParam.type = tuning;
tuningParam.sigma = sigma;
tuningParam.H = paramH;
tuningParam.G = paramG;

for songloc = songLocs
    close all
    maskerloc=0;
    t_spiketimes=InputGaussianSTRF_v2(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maskerlvl);
    t_spiketimes=InputGaussianSTRF_v2(maskerloc,songloc,tuningParam,saveParam,mean_rate,stimGain,maskerlvl);
    for maskerloc = maskerLocs
        t_spiketimes=InputGaussianSTRF_v2(songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maskerlvl);
    end
end

%% Grids for each neuron
% fileloc = 'C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\ICSimStim\mouse\v2\155210_seed142307_s30';
fileloc = [saveParam.fileLoc];
allfiles = dir([fileloc filesep '*.mat'])
tgtalone = dir([fileloc filesep '*m0.mat'])
mskalone = dir([fileloc filesep 's0*.mat'])
mixedfiles = setdiff({allfiles.name},[{tgtalone.name};{mskalone.name}])
for i = 1:16
    data = load([fileloc filesep mixedfiles{i}]);
    perf(i,:) = data.disc;
end

neurons = {'left sigmoid','gaussian','u','right sigmoid'};
[X,Y] = meshgrid(songLocs,fliplr(maskerLocs));
figure;
for i = 1:length(neurons)
    subplot(2,2,i)
    neuronPerf = perf(:,i);
    str = cellstr(num2str(round(neuronPerf)));
    neuronPerf = reshape(neuronPerf,4,4);
    imagesc(flipud(neuronPerf));
    colormap('parula');
    xticks([1:4]); xticklabels({'-90','0','45','90'})
    yticks([1:4]); yticklabels(fliplr({'-90','0','45','90'}))
    title(neurons(i))
    text(X(:)-0.2,Y(:),str,'Fontsize',12)
    caxis([50,100])
    xlabel('Song Location')
    ylabel('Masker Location')
    set(gca,'fontsize',12)
end
saveas(gca,[fileloc filesep 'performance_grid.tiff'])
end