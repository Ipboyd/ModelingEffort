% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

% note:
% large bottleneck lies in r/w to network drive

clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('..\genlib')
addpath('..\stimuli')
% dataSaveLoc = 'Z:\eng_research_hrc_binauralhearinglab\kfchou\ActiveProjects\MiceSpatialGrids\ICStim';
dataSaveLoc = pwd; %local save location

% Spatial tuning curve parameters
sigma = 30; %60 for bird but 38 for mouse
tuning = 'mouse'; %'bird' or 'mouse'
stimGain = 0.5;
maskerlvl = 0.01; %default is 0.01
maxWeight = 1; %maximum mixed tuning weight; capped at this level.
tic;

% load stimuli & calc spectrograms
if strcmp(tuning,'mouse')
    [song1,~] = audioread('200k_target1.wav');
    [song2,~] = audioread('200k_target2.wav');
    for trial = 1:10
        [masker,fs] = audioread(['200k_masker' num2str(trial) '.wav']);
        [spec,~,~]=STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
        masker_specs{trial} = spec;
    end
    
    %strf parameters
    paramH.BW= 0.05; %bandwidth
    paramH.BTM= 3.8 ; %best temporal modulation
    paramH.t0= 0.1; % t0, peak latency (s)
    paramH.phase= 0.48*pi; % phase
    paramG.BW=2000;  % Hz
    paramG.BSM=5.00E-05; % 1/Hz=s best spectral modulation
    paramG.f0=4300;
    strfGain = 0.35;
elseif strcmp(tuning,'bird')
    % stimuli
    load('stimuli_birdsongs.mat','stimuli','fs')
    minlen = min(cellfun(@length,stimuli));
    song1 = stimuli{1}(1:minlen);
    song2 = stimuli{2}(1:minlen);
    masker = stimuli{3}(1:minlen);
    [spec,~,~]=STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
    for trial = 1:10
        masker_specs{trial} = spec;
    end
    % STRF parameters from Junzi's simulations
    load('bird_STRF_params.mat');
end
[song1_spec,t,f]=STRFspectrogram(song1/rms(song1)*0.01,fs);
[song2_spec,~,~]=STRFspectrogram(song2/rms(song2)*0.01,fs);
specs.songs{1} = song1_spec;
specs.songs{2} = song2_spec;
specs.maskers = masker_specs;
specs.dims = size(song1_spec);
specs.t = t;
specs.f = f;

% make STRF
strf=STRFgen(paramH,paramG,f,t(2)-t(1));
strf.w1 = strf.w1*strfGain;
% ============ log message (manual entry?) ============
saveName = sprintf('full_grids\\BW_%0.3f BTM_3.8 t0_0.1 phase%0.4f\\s%d_STRFgain%0.2f_%s',...
                paramH.BW,paramH.phase/pi,sigma,strfGain,datestr(now,'YYYYmmdd-HHMMSS'));
saveFlag = 0;

msg{1} = ['capped tuning weight to' num2str(maxWeight)];
msg{end+1} = ['maskerlvl = ' num2str(maskerlvl)];
msg{end+1} = ['strfGain = ' num2str(strfGain)];
msg{end+1} = ['strf paramH.BW = ' num2str(paramH.BW)];
msg{end+1} = ['strf paramH.BTM = ' num2str(paramH.BTM)];
msg{end+1} = ['strf paramH.t0 = ' num2str(paramH.t0)];
msg{end+1} = ['strf paramH.phase = ' num2str(paramH.phase)];
msg{end+1} = ['strf paramG.BW= ' num2str(paramG.BW)];
% =============== end log file ===================

%% Run simulation script
mean_rate = 0.1;
songLocs = 1:4;
maskerLocs = 1:4;

saveParam.flag = 1;
saveParam.fileLoc = [dataSaveLoc filesep tuning filesep saveName];
if ~exist(saveParam.fileLoc,'dir'), mkdir(saveParam.fileLoc); end
tuningParam.strf = strf;
tuningParam.type = tuning;
tuningParam.sigma = sigma;

% iterate over all location combinations
set(0, 'DefaultFigureVisible', 'off')
figure;
for songloc = songLocs
    close all
    maskerloc=0;
    
    t_spiketimes=InputGaussianSTRF_v2(specs,songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
    t_spiketimes=InputGaussianSTRF_v2(specs,maskerloc,songloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
    for maskerloc = maskerLocs
        t_spiketimes=InputGaussianSTRF_v2(specs,songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
    end
end
set(0, 'DefaultFigureVisible', 'on')

% write log file
msg{end+1} = ['elapsed time is ' num2str(toc) ' seconds'];
fid = fopen(fullfile(saveParam.fileLoc, 'notes.txt'), 'a');
if fid == -1
  error('Cannot open log file.');
end
for k=1:length(msg), fprintf(fid, '%s: %s\n', datestr(now, 0), msg{k}); end
fclose(fid);
%% Grids for each neuron
% fileloc =
% 'C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\ICSimStim\mouse\v2\155210_seed142307_s30'; dataloc?
% fileloc = 'Z:\eng_research_hrc_binauralhearinglab\kfchou\ActiveProjects\MiceSpatialGrids\ICStim\Mouse\s30_gain0.5_maskerLvl0.01_20200415-213511';
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


