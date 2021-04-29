% Experiment data -> DNN training Target 
%
% take spiketimes obtained from experiments, convert to spike trains used
% by the model. The resulting matrix has the dimensions [trials x time]
% The spikes along the time axis are organized as follows:
%   s1m0, s2m0, s3m0, s4m0, s1m1, s1m2, s1m3, s1m4, s2m1, ...
%
% 
%
addpath('plotting')
subjectID = '616283';
TMR = '6';
laserPow = '2';
dataLoc = ['Z:\eng_research_hrc_binauralhearinglab\noconjio\Spatial grid database\Subjects with opto and TMR sessions (Fall 2020 - Present)\' subjectID '\' subjectID '-' laserPow 'mW_75dBtarget-' TMR 'dBTMR_Kilosort_correctMap\No Laser'];
rawData = [subjectID '_' laserPow 'mW_75dBtarget_' TMR 'dBTMR_raw(-1,4)_Kilosort.mat'];
perfData = [subjectID '-' laserPow 'mW_' TMR 'dBTMR_performance.mat'];
load([dataLoc filesep rawData],'Spks_clean','Spks_masked')
load([dataLoc filesep perfData],'clean_performance','masked_performance')

channel_to_model = 27;
numTrialsTotal = 20;
z = 1;
debug_flag = 0;


tBeforeStim = 0; %ms
% temp1 = cellfun(@max,Spks_masked{channel_to_model},'UniformOutput',false);
% temp2 = cellfun(@max,Spks_clean{channel_to_model},'UniformOutput',false);
% temp = [temp1(:); temp2(:)];
% tmax = max([temp{:}])/dt+tBeforeStim;
% tmax = round(tmax);


%% Define Kernel for smoothing spike trains
dt = 0.1;
t = (0:dt:500); % 0-500ms
tau = 5; % ms
kernel = t.*exp(-t/tau);

%% calculate max lag
load('data_recordings\PSTH_IC_clean.mat')
% for loc = 1:4
% [r,lags] = xcorr(psth1clean{1},psths1{loc,channel_to_model});
% maxR(loc) = max(r);
% maxLag(loc) = lags(r==max(r));
% 
% [r,lags] = xcorr(psth2clean{1},psths2{loc,channel_to_model});
% maxR2(loc) = max(r);
% maxLag2(loc) = lags(r==max(r));
% end
% round([maxLag', maxLag2'])
%% clean conditions
dt = 0.1; % seconds
dt = dt/1000; % milliseconds
tmax = 3/dt;
padToTime = round(tmax+2000);
spks = [];
% s1cutoffs = [2510,30200];
% s2cutoffs = [2510,30170];
       
% s1delay = round(mean(maxLag(1,:)));
% s2delay = round(mean(maxLag2(1,:)));
s1delay = 0;
s2delay = 0;
for chan = channel_to_model
    if isempty(Spks_clean{chan}), continue; end
    
    for loc = 1:4
        singleConfigSpks = zeros(20,tmax);
        for j = 1:10
            for song = 1:2
                spkTimes = round(Spks_clean{chan}{j,loc,song}/dt)+tBeforeStim;
                singleConfigSpks(j+(song-1)*10,spkTimes(spkTimes>0)) = 1;
            end
        end

        trialStartTimes(z) = padToTime;
        z = z+1;
        % pad to compensate for delay
        singleConfigSpks(1:10,:) = [zeros(10,s1delay), singleConfigSpks(1:10,1:end-s1delay)];
        singleConfigSpks(11:20,:) = [zeros(10,s2delay), singleConfigSpks(11:20,1:end-s2delay)];
        
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,2) < padToTime
            padSize = padToTime-size(singleConfigSpks,2);
            singleConfigSpks = cat(2,singleConfigSpks,zeros(20,padSize));
        elseif size(singleConfigSpks,2) > padToTime
            singleConfigSpks = cat(2,singleConfigSpks(:,1:tmax),zeros(20,padToTime-tmax));
        end
        
        % zero out signals where input signal is 0
%         singleConfigSpks(1:10,1:s1cutoffs(1)) = 0;
%         singleConfigSpks(1:10,s1cutoffs(2):end) = 0;
%         singleConfigSpks(11:20,1:s2cutoffs(1)) = 0;
%         singleConfigSpks(11:20,s2cutoffs(2):end) = 0;
        
        % concatenate
        spks = cat(2,spks,singleConfigSpks);

        summedSpks = sum(singleConfigSpks(1:10,:));
        smoothed1 = conv(summedSpks,kernel);
        psths1{loc,chan} = smoothed1;

        summedSpks = sum(singleConfigSpks(11:20,:));
        smoothed2 = conv(summedSpks,kernel);
        psths2{loc,chan} = smoothed2;
        if debug_flag
            figure;
            subplot(2,2,1)
            plotSpikeRasterFs(logical(singleConfigSpks(1:10,:)),'PlotType','vertline2');
            xlim([0 tmax])

            subplot(2,2,3)
            plot(smoothed1);
            xlim([0 tmax])

            subplot(2,2,2)
            plotSpikeRasterFs(logical(singleConfigSpks(11:20,:)),'PlotType','vertline2');
            xlim([0 tmax])

            subplot(2,2,4)
            plot(smoothed2);
            xlim([0 tmax])

            sgtitle(['location index # ' num2str(loc)])
        end
    end
end

% check for alignment with input
figure;
loc = 1;
subplot(2,1,1)
plot(psth1clean{loc}); hold on;
subplot(2,1,2)
plot(psth2clean{loc}); hold on;
        
subplot(2,1,1)
plot(psths1{loc,chan}); hold on;
subplot(2,1,2)
plot(psths2{loc,chan}); hold on;

% ====== calculate performance ======
% % % spks to spiketimes in a cell array of 20x2
% numTrials = 20; %total trials over 2 songs
% tau = linspace(1,30,100);
% spkTime = cell(numTrials,1);
% for ii = 1:numTrials, spkTime{ii} = find(singleConfigSpks(ii,:)); end
% spkTime = reshape(spkTime,numTrials/2,2);
% % calculate distance matrix & performance
% distMat = calcvr(spkTime, tau);
% [performance, ~] = calcpc(distMat, numTrials/2, 2, 1,[], 'new');
% pc = mean(max(performance));
% PCstr = ['PC = ' num2str(pc)]

%% masked conditions
for k = 1:4 %target loc (1:4)
    for l = 1:4 %masker loc (1:4)
        singleConfigSpks = zeros(numTrialsTotal,round(tmax)); %I'm storing spikes in a slightly different way...
        for j = 1:10 %trials [1:10]
            for song = 1:2
                spkTimes = round(Spks_masked{channel_to_model}{j,k,l,song}/dt)+tBeforeStim;
                singleConfigSpks(j+(song-1)*10,spkTimes(spkTimes>0)) = 1;
            end
        end
%         figure;
%         plotSpikeRasterFs(logical(singleConfigSpks),'PlotType','vertline2');
%         xlim([0 tmax])

        trialStartTimes(z) = padToTime;
        z = z+1;
        
        % pad to compensate for delay
        singleConfigSpks(1:10,:) = [zeros(10,s1delay), singleConfigSpks(1:10,1:end-s1delay)];
        singleConfigSpks(11:20,:) = [zeros(10,s2delay), singleConfigSpks(11:20,1:end-s2delay)];
        
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,2) < padToTime
            padSize = padToTime-size(singleConfigSpks,2);
            singleConfigSpks = cat(2,singleConfigSpks,zeros(20,padSize));
        elseif size(singleConfigSpks,2) > padToTime
            singleConfigSpks = cat(2,singleConfigSpks(:,1:tmax),zeros(20,padToTime-tmax));
        end
        
        % zero out signals where input signal is 0
%         singleConfigSpks(1:10,1:s1cutoffs(1)) = 0;
%         singleConfigSpks(1:10,s1cutoffs(2):end) = 0;
%         singleConfigSpks(11:20,1:s2cutoffs(1)) = 0;
%         singleConfigSpks(11:20,s2cutoffs(2):end) = 0;
        
        % concatenate
        spks = cat(2,spks,singleConfigSpks);
        
        if debug_flag
            figure;
            subplot(2,2,1)
            plotSpikeRasterFs(logical(singleConfigSpks(1:10,:)),'PlotType','vertline2');
            xlim([0 tmax])

            subplot(2,2,3)
            summedSpks = sum(singleConfigSpks(1:10,:));
            smoothed = conv(summedSpks,kernel);
            plot(smoothed);
            xlim([0 tmax])

            subplot(2,2,2)
            plotSpikeRasterFs(logical(singleConfigSpks(11:20,:)),'PlotType','vertline2');
            xlim([0 tmax])

            subplot(2,2,4)
            summedSpks = sum(singleConfigSpks(11:20,:));
            smoothed = conv(summedSpks,kernel);
            plot(smoothed);
            xlim([0 tmax])

            sgtitle(sprintf('target loc %i masker loc %i',k,l))
        end
    end
end

%% Smooth spike trains

% sum spikes over each song
summedSpks = [sum(spks(1:10,:)), sum(spks(11:20,:))];
smoothed = conv2(summedSpks',kernel');
plot(smoothed)

% save as output
output_training = smoothed(1:length(summedSpks));
saveFile = ['data_recordings\' subjectID filesep TMR 'TMR_' laserPow 'mW'];
if ~exist(saveFile,'dir'), mkdir(saveFile); end
save([saveFile filesep 'training_output_chan27.mat'],'output_training');


% % %% do the same for IC input
% % % run the first 3 cells of genTrainingData
% % % use subz = [[5,10,15,20],[6:9,11:14,16:19,21:24]];
% % % they load the IC spikes and configures them in the correct way.
% % psth_in = cat(3,sum(spks(1:10,:,:)), sum(spks(11:20,:,:)));
% % psth_in  = squeeze(psth_in);
% % smoothed_input = conv2(psth_in,kernel');
% % input_training = smoothed_input;
% % % save('data_recordings\616305\training_input.mat','input_training');

%% plot performance
best_tau = 5;
% channel_to_model = 30;
cleanPerf = clean_performance{channel_to_model}(best_tau,:)
mixedPerf = squeeze(masked_performance{channel_to_model}(best_tau,:,:))

figure;
[X,Y] = meshgrid(1:4,1:4);
subplot('position',[0.1 0.75 0.8 0.15])
imagesc(cleanPerf)
text((1:4)-0.25,ones(1,4),num2str(cleanPerf'))
xticklabels([])
yticklabels([])
caxis([50 100])
subplot('position',[0.1 0.1 0.8 0.6])
imagesc(mixedPerf)
xticks([1,2,3,4])
xticklabels({'90','45','0','-90'})
yticks([1,2,3,4])
yticklabels({'-90','0','45','90'})
ylabel('masker location')
xlabel('song location')
text(X(:)-0.25,Y(:),num2str(mixedPerf(:)))
caxis([0 100])