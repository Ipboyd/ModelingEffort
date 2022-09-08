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
sigma = 38; %60 for bird but 38 for mouse
tuning = 'mouse';
stimGain = 0.5;
targetlvl = 0.01;
maskerlvl = 0.01; % default is 0.01
maxWeight = 1; % maximum mixed tuning weight; capped at this level.
tic;

% load stimuli & calc spectrograms
%     [song1,~] = audioread('200k_target1.wav');
%     [song2,~] = audioread('200k_target2.wav');

%     for trial = 1:10
%         [masker,fs] = audioread(['200k_masker' num2str(trial) '.wav']);
%         [spec,~,~] = STRFspectrogram(masker/rms(masker)*maskerlvl,fs);
%         masker_specs{trial} = spec;
%     end

% [song1_spec,t,f]=STRFspectrogram(song1/rms(song1)*targetlvl,fs);
% [song2_spec,~,~]=STRFspectrogram(song2/rms(song2)*targetlvl,fs);
% specs.songs{1} = song1_spec;
% specs.songs{2} = song2_spec;
% specs.maskers = masker_specs;
% specs.dims = size(song1_spec);
% specs.t = t;
% specs.f = f;

paramH.alpha = 0.0097; % time constant of temporal kernel [s] 0.0097
paramH.N1 = 5;
paramH.N2 = 7;
paramH.SC1 = 1;
paramH.SC2 = 0.88;  % increase -> more inhibition

paramG.BW = 2000;  % Hz
paramG.BSM = 5.00E-05; % 1/Hz=s best spectral modulation
paramG.f0 = 4300; % ~strf.f(30)
strfGain = 0.10;
load('preprocessed_stims.mat'); % don't need to run STRFspectrogram again

% make STRF
strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1));
strf.w1 = strf.w1*strfGain;

% ============ log message ============

paramSpk.t_ref = 1;
paramSpk.t_ref_rel = 1;
paramSpk.rec = 2;
paramSpk.offsetFrac = 1;

saveName = sprintf('inhib-input-to-center-V3//alpha_%0.3f//STRFgain %0.2f, %0.1fms tau_rel',...
    paramH.alpha,strfGain,paramSpk.t_ref_rel);
saveFlag = 0;

msg{1} = ['capped tuning weight to ' num2str(maxWeight)];
msg{end+1} = ['maskerlvl = ' num2str(maskerlvl)];
msg{end+1} = ['strfGain = ' num2str(strfGain)];
msg{end+1} = ['offsetRateFrac = ' num2str(paramSpk.offsetFrac)];
msg{end+1} = ['absolute refractory period = ' num2str(paramSpk.t_ref)];
msg{end+1} = ['relative refractory period = ' num2str(paramSpk.t_ref_rel)];
msg{end+1} = ['recovery steepness = ' num2str(paramSpk.rec)];
msg{end+1} = ['strf paramH.alpha = ' num2str(paramH.alpha)];
msg{end+1} = ['strf paramH.N1 = ' num2str(paramH.N1)];
msg{end+1} = ['strf paramH.N2 = ' num2str(paramH.N2)];
msg{end+1} = ['strf paramH.SC1 = ' num2str(paramH.SC1)];
msg{end+1} = ['strf paramH.SC2 = ' num2str(paramH.SC2)];
msg{end+1} = ['strf paramG.BW = ' num2str(paramG.BW)];

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
figure;

for tloc = songLocs
    close all

    inputEI(specs,tloc,0,tuningParam,saveParam,mean_rate,stimGain,maxWeight,paramSpk);
    inputEI(specs,0,tloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight,paramSpk);
    for mloc = maskerLocs
        inputEI(specs,tloc,mloc,tuningParam,saveParam,mean_rate,stimGain,maxWeight,paramSpk);
    end
end

% write log file
msg{end+1} = ['elapsed time is ' num2str(toc) ' seconds'];
fid = fopen(fullfile(saveParam.fileLoc, 'notes.txt'), 'a');
if fid == -1
    error('Cannot open log file.');
end
for k=1:length(msg), fprintf(fid, '%s: %s \n ', datestr(now, 0), msg{k}); end
fclose(fid);

%% Grids for each neuron

fileloc = saveParam.fileLoc;
addpath('..')

% fileloc = [saveParam.fileLoc];
allfiles = dir([fileloc filesep '*.mat']);
tgtalone = dir([fileloc filesep '*m0.mat']);
mskalone = dir([fileloc filesep 's0*.mat']);
mixedfiles = setdiff({allfiles.name},[{tgtalone.name};{mskalone.name}]);

% call mixed configs and build grids for each channel
clear perf fr avgFR
for i = 1:16
    data = load([fileloc filesep mixedfiles{i}]);
    perf(i,:) = data.disc;
    avgFR(i,:) = data.avgSpkRate_on;
end

neurons = fliplr({'ipsi sigmoid','gaussian','u','contra sigmoid'});
% neurons = {'-90','0','45','90'};
figure('position',[50 400 1600 400]);
left = 0.05;
bottom = 0.15;
width = 0.24;
height = 0.6;

% plot mixed grid
for i = 1:length(neurons)
    subplot('position',[left+width*(i-1) bottom width*0.8 height])
    neuronPerf = perf(:,i);
    plotPerfGrid(neuronPerf,avgFR(:,i),'');
    xticks([1:4]); xticklabels(fliplr({'-90','0','45','90'}))
    yticks([1:4]); yticklabels({'-90','0','45','90'})
    xlabel('Song Location')
    ylabel('Masker Location')
end

% target only, masker only configs
clear perf fr avgFR
for i = 1:4 % per location
    data = load([fileloc filesep tgtalone(i).name]);
    perf(:,i,1) = data.disc;
    avgFR(:,i,1) = data.avgSpkRate_on;

    data = load([fileloc filesep mskalone(i).name]);
    perf(:,i,2) = data.disc;
    avgFR(:,i,2) = data.avgSpkRate_on;
end
left = 0.05;
bottom = 0.76;
width = 0.24;
height = 0.15;
for i = 1:length(neurons)
    subplot('position',[left+width*(i-1) bottom width*0.8 height])
    neuronPerf = squeeze(perf(i,:,:))';
    plotPerfGrid(neuronPerf,squeeze(avgFR(i,:,:))',neurons(i));
end
saveas(gca,[fileloc filesep 'performance_grid.tiff'])


