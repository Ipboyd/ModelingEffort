% Experiment data -> DNN training Input
%
% spikes -> PSTH -> weight by tuning curve -> spikes
%
%
% Need to manually determine
% 1. representative PSTH for song
% 2. representative PSTH for noise

addpath('plotting')
addpath('data_recordings')

% experiment data to use
subjectID = '616283';
TMR = '6';
laserPow = '2';
expChannel = 27;
songPSTHlocIdx = 1;
noisePSTHsongloc = 3;
noisePSTHmaskloc = 3;

% load data to use as input
dataLoc = ['Z:\eng_research_hrc_binauralhearinglab\noconjio\Spatial grid database\Subjects with opto and TMR sessions (Fall 2020 - Present)\' subjectID '\' subjectID '-' laserPow 'mW_75dBtarget-' TMR 'dBTMR_Kilosort_correctMap\No Laser'];
rawData = [subjectID '_' laserPow 'mW_75dBtarget_' TMR 'dBTMR_raw(-1,4)_Kilosort.mat'];
perfData = [subjectID '-' laserPow 'mW_' TMR 'dBTMR_performance.mat'];
load([dataLoc filesep rawData],'Spks_clean','Spks_masked')
load([dataLoc filesep perfData],'clean_performance','masked_performance')

%% mouse tuning curves
addpath('ICSimStim')
sigma = 30;
x=-108:108;
tuningcurve=zeros(4,length(x));
tuningcurve(1,:)= sigmf(-x,[0.016 -22.5])-0.05; %flipped sigmodial
tuningcurve(2,:)=gaussmf(x,[sigma,0]); %guassian
tuningcurve(3,:)= 1- gaussmf(x,[sigma,0]); %U shaped gaussian
tuningcurve(4,:)= sigmf(x,[0.016 -22.5])-0.05; %sigmodial
neuronNames = {'left sigmoid','gaussian','U','right sigmoid'};
azimuth=[-90 0 45 90]; %stimuli locations

%% Define Kernel for smoothing spike trains
dt = 0.1;
t = (0:dt:500); % 0-500ms
tau = 5; % ms
kernel = t.*exp(-t/tau);

%% clean conditions - extract experimental data between 0 and 3 seconds, calculate PSTH
channel_to_model = expChannel;
dt = 0.1; % seconds
dt = dt/1000; % milliseconds
tmax = 3/dt;
tBeforeStim = 0;
padToTime = round(tmax+2000);
for chan = channel_to_model
    if isempty(Spks_clean{chan}), continue; end
    
    for loc = songPSTHlocIdx
        singleConfigSpks = zeros(20,tmax);
        for j = 1:10
            for song = 1:2
                spkTimes = round(Spks_clean{chan}{j,loc,song}/dt)+tBeforeStim;
                singleConfigSpks(j+(song-1)*10,spkTimes(spkTimes>0)) = 1;
            end
        end

        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,2) < padToTime
            padSize = padToTime-size(singleConfigSpks,2);
            singleConfigSpks = cat(2,singleConfigSpks,zeros(20,padSize));
        elseif size(singleConfigSpks,2) > padToTime
            singleConfigSpks = cat(2,singleConfigSpks(:,1:tmax),zeros(20,padToTime-tmax));
        end
        
        % psth
        summedSpks = sum(singleConfigSpks(1:10,:));
        smoothed1 = conv(summedSpks,kernel);
        psths1{loc} = smoothed1(1:length(summedSpks));
        summedSpks = sum(singleConfigSpks(11:20,:));
        smoothed2 = conv(summedSpks,kernel);
        psths2{loc} = smoothed2(1:length(summedSpks));
    end
end
% normalize PSTH by its mean
PSTH_song{1} = psths1{1}/mean(psths1{1}); 
PSTH_song{2} = psths2{1}/mean(psths2{1});

%% noise conditions  - extract experimental data between 0 and 3 seconds, calculate PSTH
dt = 0.1; % seconds
dt = dt/1000; % milliseconds
tmax = 3/dt;
tBeforeStim = 0;
padToTime = round(tmax+2000);
for k = noisePSTHsongloc  %target loc (1:4)
    for l = noisePSTHmaskloc  %masker loc (1:4)
        singleConfigSpks = zeros(20,round(tmax)); %I'm storing spikes in a slightly different way...
        for j = 1:10 %trials [1:10]
            for song = 1:2
                spkTimes = round(Spks_masked{channel_to_model}{j,k,l,song}/dt)+tBeforeStim;
                singleConfigSpks(j+(song-1)*10,spkTimes(spkTimes>0)) = 1;
            end
        end

        figure;
        plotSpikeRasterFs(logical(singleConfigSpks),'PlotType','vertline2');
        xlim([0 tmax])

        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,2) < padToTime
            padSize = padToTime-size(singleConfigSpks,2);
            singleConfigSpks = cat(2,singleConfigSpks,zeros(20,padSize));
        elseif size(singleConfigSpks,2) > padToTime
            singleConfigSpks = cat(2,singleConfigSpks(:,1:tmax),zeros(20,padToTime-tmax));
        end
        
        % psth
        summedSpks = sum(singleConfigSpks(1:10,:));
        smoothed1 = conv(summedSpks,kernel);
        psths1{k,l} = smoothed1(1:length(summedSpks));
        summedSpks = sum(singleConfigSpks(11:20,:));
        smoothed2 = conv(summedSpks,kernel);
        psths2{k,l} = smoothed2(1:length(summedSpks));
        
        
    end
end
% normalize PSTH by its mean
PSTH_noise{1} = psths1{k,l}/mean(psths1{1}); 
PSTH_noise{2} = psths2{k,l}/mean(psths2{1});

%% check PSTH
figure;
plot(PSTH_song{2}); hold on;
plot(PSTH_noise{2})
%% Weight song and noise PSTHs by tuning curves; combine
maxWeight = 1;
stimGain = 0.75;

for songn = 1:2
    songPSTH = PSTH_song{songn};
    for songloc = 1:4
        maskerloc = 0;
        for locIdx = 1:4 % "spatial" channel
            if songloc, songWeight = tuningcurve(locIdx,x==azimuth(songloc)); end
            PSTH_clean{songn,songloc}(:,locIdx) = songPSTH*songWeight;
        end
    end
end

for maskerloc = 1:4
    for songloc = 1:4

        for songn = 1:2
            songPSTH = PSTH_song{songn};
            masker = PSTH_noise{songn};
            % weight input PSTHs & noise by tuning curves
            for locIdx = 1:4 % "spatial" channel
                maskerWeight = 0;
                songWeight = 0;

                % 1. weight stimulus at each location by tuning curve amplitude
                % 2. sum weighted stimuli
                if maskerloc, maskerWeight = tuningcurve(locIdx,x==azimuth(maskerloc)); end
                if songloc, songWeight = tuningcurve(locIdx,x==azimuth(songloc)); end
                totalWeight = maskerWeight + songWeight;

                mixed_rate = masker*maskerWeight + songPSTH*songWeight;

                % cap total weight to maxWeight, then scale
                if totalWeight > maxWeight
                    mixed_rate = mixed_rate/totalWeight*maxWeight;
                end
                mixed_rate = mixed_rate * stimGain;
                PSTH_mixed{songn,songloc,maskerloc}(:,locIdx) = mixed_rate;
%                 end
            end
        end
    end
end

%% reformat for DNN
inputPSTH = [];
for songn = 1:2
    for songloc = 1:4
        inputPSTH = cat(1,inputPSTH,PSTH_clean{songn,songloc});
    end
    for songloc = 1:4
        for maskerloc = 1:4
            inputPSTH = cat(1,inputPSTH,PSTH_mixed{songn,songloc,maskerloc});
        end
    end
end

% save training input
input_training = inputPSTH;
saveFile = ['data_recordings\' subjectID filesep TMR 'TMR_' laserPow 'mW'];
if ~exist(saveFile,'dir'), mkdir(saveFile); end
save([saveFile filesep 'training_input_chan27.mat'],'input_training');

%% generate spikes from PSTHs, save results
% dt = 0.0001 seconds
% returned spike_times = milliseconds
% tau = milliseconds
addpath('genlib')
channelNames = {'left sigmoid','gauss','U','right sigmoid'};
numTrials = 10;
locLabels = {'90','45','0','-90'};
tau = linspace(1,30,100);
mean_rate = 30; %hz

% calculate spike times
for songloc = 1:4
    ICspks_clean(songloc).spks = zeros(20,4,3000);

    for channel = 1:4
        tempSpkTimes = [];
        for songn = 1:2
            [spike_times, spkcnt] = PSTH2Spks(PSTH_clean{songn,songloc}(:,channel),numTrials,mean_rate,dt);
            tempSpkTimes = [tempSpkTimes, spike_times];
        end
        % calculate distance matrix & performance
        % spkTime has dimensions of nTrials x nSong
        
        distMat = calcvr(tempSpkTimes, tau);
        [performance, ~] = calcpc(distMat, numTrials, 2, 1,[], 'new');
        
        % convert to spike trains...
        ICspks_clean(songloc).perf(channel) = max(performance);
        for ii = 1:10
            for jj = 1:2
                currentSpkTimes = round(tempSpkTimes{ii,jj});
                ICspks_clean(songloc).spks(ii + 10*(jj-1),channel,currentSpkTimes) = 1;
            end
        end    
    end
end

for songloc = 1:4
    for maskloc = 1:4
        ICspks_mixed(songloc,maskloc).spks = zeros(20,4,3/dt);

        for channel = 1:4
            tempSpkTimes = [];
            for songn = 1:2
                [spike_times, spkcnt] = PSTH2Spks(PSTH_mixed{songn,songloc,maskloc}(:,channel),numTrials,mean_rate,dt);
                tempSpkTimes = [tempSpkTimes, spike_times];
            end
             distMat = calcvr(tempSpkTimes, tau);
            [performance, ~] = calcpc(distMat, numTrials, 2, 1,[], 'new');

            % convert to spike trains...
            ICspks_mixed(songloc,maskloc).perf(channel) = max(performance);
            for ii = 1:10
                for jj = 1:2
                    currentSpkTimes = round(tempSpkTimes{ii,jj});
                    ICspks_mixed(songloc,maskloc).spks(ii + 10*(jj-1),channel,currentSpkTimes) = 1;
                end
            end
            
        end
    end
end

% plot
for channel = 1:4
    figure('position',[100 100 900 800])
    for songloc = 1:4
        currentSpks = squeeze(ICspks_clean(songloc).spks(:,channel,:));
        subplot('position',[0.075+0.22*(songloc-1) 0.7 0.2 0.175])
        plotSpikeRasterFs(logical(currentSpks),'PlotType','vertline2');
        xticks([])
        yticks([])
        xlim([0 3000])
        title(['performance: ' num2str(ICspks_clean(songloc).perf(channel))])
    end
    
    for songloc = 1:4
        for maskloc = 1:4
            currentSpks = squeeze(ICspks_mixed(songloc,maskloc).spks(:,channel,:));
            subplot('position',[0.075+0.22*(songloc-1) 0.075+0.15*(maskloc-1) 0.2 0.125])
            plotSpikeRasterFs(logical(currentSpks),'PlotType','vertline2');
            xticks([])
            yticks([])
            xlim([0 3000])
            title(['performance: ' num2str(ICspks_mixed(songloc,maskloc).perf(channel))])

            if maskloc == 1
                xlabel(['songloc' locLabels(songloc)])
            end
            if songloc == 1
                ylabel(['maskloc' locLabels(maskloc)])
            end
        end
    end
    sgtitle(channelNames{channel})
end

%% save data
meta.subjectID = subjectID;
meta.TMR = TMR;
meta.laserPow = laserPow;
meta.expChannel = expChannel;
meta.songPSTHlocIdx = 1;
meta.noisePSTHsongloc = 3;
meta.noisePSTHmaskloc = 3;
meta.dataLoc = dataLoc;
meta.rawDataFile = rawData;
meta.perfDataFile = perfData;

params.sigma = sigma;
params.tuningCurve = tuningcurve;
params.kernel.tau = tau;
params.kernel.y = kernel;

data.PSTH.song = PSTH_song;
data.PSTH.noise = PSTH_noise;
data.PSTH.mixed = PSTH_mixed;

filename = sprintf('data_recordings\\reconfigured\\%s %sTMR %sdB chan%i.mat',subjectID,TMR,laserPow,expChannel);
save(filename,'ICspks_mixed','ICspks_clean','meta','params','data')
% PSTH_clean
% PSTH_mixed