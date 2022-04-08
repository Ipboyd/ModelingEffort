% Simulate spike times based on a model neuron with a (1) STRF and a (2)
% spatial tuning curve. I.e. this neuron has both spatial and
% spectral-temporal tuning.

% note:
% large bottleneck lies in r/w to network drive

cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim');
clearvars;clc;close all
addpath(genpath('strflab_v1.45'))
addpath('../genlib')
addpath('../fixed-stimuli')
addpath('../plotting')
addpath('../cSPIKE')
InitializecSPIKE;
% dataSaveLoc = 'Z:/eng_research_hrc_binauralhearinglab/kfchou/ActiveProjects/MiceSpatialGrids/ICStim';
dataSaveLoc = pwd; %local save location

% Spatial tuning curve parameters
sigma = 38; %60 for bird but 38 for mouse
tuning = 'mouse'; %'bird' or 'mouse'
stimGain = 0.5;
targetlvl = 0.01;
maskerlvl = 0.01; %default is 0.01
maxWeight = 1; %maximum mixed tuning weight; capped at this level.
tic;

% load stimuli & calc spectrograms
if strcmp(tuning,'mouse')
%     [song1,~] = audioread('200k_target1.wav');
%     [song2,~] = audioread('200k_target2.wav');
    
%     for trial = 1:10
%         [masker,fs] = audioread(['200k_masker' num2str(trial) '.wav']);
%         [spec,~,~] = STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
%         masker_specs{trial} = spec;
%     end
    
    paramH.alpha = 0.0097; % time constant of temporal kernel [s]
    paramH.N1 = 5;
    paramH.N2 = 7;
    paramH.SC1 = 1;
    paramH.SC2 = 0.88;  % increase -> more inhibition
    
    paramG.BW = 2000;  % Hz
    paramG.BSM = 5.00E-05; % 1/Hz=s best spectral modulation
    paramG.f0 = 4300; % ~strf.f(30)
    strfGain = 0.17;
    load('preprocessed_stims.mat'); % don't need to run STRFspectrogram again
    
elseif strcmp(tuning,'bird')
    % stimuli
    load('stimuli_birdsongs.mat','stimuli','fs')
%     minlen = min(cellfun(@length,stimuli));
%     song1 = stimuli{1}(1:minlen);
%     song2 = stimuli{2}(1:minlen);
%     masker = stimuli{3}(1:minlen);
%     [spec,~,~]=STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
%     for trial = 1:10
%         masker_specs{trial} = spec;
%     end
    % STRF parameters from Junzi's simulations
    load('bird_STRF_params.mat');
end

% [song1_spec,t,f]=STRFspectrogram(song1/rms(song1)*targetlvl,fs);
% [song2_spec,~,~]=STRFspectrogram(song2/rms(song2)*targetlvl,fs);
% specs.songs{1} = song1_spec;
% specs.songs{2} = song2_spec;
% specs.maskers = masker_specs;
% specs.dims = size(song1_spec);
% specs.t = t;
% specs.f = f;

offsetFrac = 1;


% make STRF
% strf=STRFgen(paramH,paramG,f,t(2)-t(1));
strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));
strf.w1 = strf.w1*strfGain;

saveName = sprintf('current//alpha_%0.3f N1_%0.0f N2_%0.0f//s%d_STRFgain%0.2f_%s',...
                paramH.alpha,paramH.N1,paramH.N2,sigma,strfGain,datestr(now,'YYYYmmdd-HHMMSS'));
saveFlag = 0;

msg{1} = ['capped tuning weight to' num2str(maxWeight)];
msg{end+1} = ['maskerlvl = ' num2str(maskerlvl)];
msg{end+1} = ['strfGain = ' num2str(strfGain)];
msg{end+1} = ['offsetRateFrac = ' num2str(offsetFrac)];
msg{end+1} = ['strf paramH.alpha = ' num2str(paramH.alpha)];
msg{end+1} = ['strf paramH.N1 = ' num2str(paramH.N1)];
msg{end+1} = ['strf paramH.N2 = ' num2str(paramH.N2)];
msg{end+1} = ['strf paramH.SC1 = ' num2str(paramH.SC1)];
msg{end+1} = ['strf paramH.SC2 = ' num2str(paramH.SC2)];
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

% % use below 2 lines to quickly check conv of STRF and spectrogram 
% strfData(song1_spec, zeros(size(song1_spec)));
% [~,resp] = linFwd_Junzi(strf);

% iterate over all location combinations
figure;
for songloc = songLocs
    close all
    maskerloc=0;
    
    InputGaussianSTRF_Current(specs,songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight,offsetFrac);
    InputGaussianSTRF_Current(specs,maskerloc,songloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight,offsetFrac);
    for maskerloc = maskerLocs
        InputGaussianSTRF_Current(specs,songloc,maskerloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight,offsetFrac);
    end
    
end

% write log file
msg{end+1} = ['elapsed time is ' num2str(toc) ' seconds'];
fid = fopen(fullfile(saveParam.fileLoc, 'notes.txt'), 'a');
if fid == -1
  error('Cannot open log file.');
end
for k=1:length(msg), fprintf(fid, '%s: %s/n', datestr(now, 0), msg{k}); end
fclose(fid);
