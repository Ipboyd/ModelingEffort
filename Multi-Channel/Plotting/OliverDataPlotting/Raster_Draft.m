
clear all
close all

load(['/Users/lbowman/Desktop/Research/all_units_info_with_polished_criteria_modified_perf.mat'],'all_data');
load(['/Users/lbowman/Desktop/Research/sound_files.mat'],'sampleRate','target1','target2');
%load(['/Users/ipboy/Downloads/all_units_info_with_polished_criteria_modified_perf.mat'],'all_data');
%load(['/Users/ipboy/Downloads/sound_files.mat'],'sampleRate','target1','target2');  

% cd(userpath);
% cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')
% 
% 
% load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
% load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

% %% plotting sound
% 
% % 
% % [audioIn, fs] = audioread('200k_target1.wav');
% % vad = voiceActivityDetector()
% % 
% % isSpeech = vad(audioIn);
% % 
% % diffSpeech = diff([0; isSpeech; 0]);
% % onsets = find(diffSpeech == 1);
% % offsets = find(diffSpeech == -1) - 1;
% % 
% % % Convert to time (seconds)
% % 
% % onsetTimes = onsets / fs;
% % offsetTimes = offsets / fs;
% cd(userpath);
% cd('../GitHub/ModelingEffort/Single-Channel/Model/PreCortical-Modeling/resampled-stimuli')
% 
% 
% % Load target 1
% [x, fs] = audioread('200k_target1.wav');
% 
% % Load target 2
% [x2, fs2] = audioread('200k_target1.wav');
% 
% 
% % Concatenate warm-up and original audio
% padded_audio = [x2; x];
% 
% % Save to new file
% audiowrite('200k_target1_with_warmup.wav', padded_audio, fs);
% 
% 
% 
% afr = dsp.AudioFileReader('200k_target1_with_warmup.wav');
% fs = afr.SampleRate;
% 
% frameSize = ceil(5e-3*fs);
% overlapSize = ceil(0.8*frameSize);
% hopSize = frameSize - overlapSize;
% afr.SamplesPerFrame = hopSize;
% 
% inputBuffer = dsp.AsyncBuffer('Capacity',frameSize);
% 
% VAD = voiceActivityDetector('FFTLength',frameSize);
% 
% scope = timescope('SampleRate',fs, ...
%     'TimeSpanSource','Property','TimeSpan',3, ...
%     'BufferLength',.5*fs, ...
%     'YLimits',[-1.5,1.5], ...
%     'TimeSpanOverrunAction','Scroll', ...
%     'ShowLegend',true, ...
%     'ChannelNames',{'Audio','Probability of speech presence'});
% 
% player = audioDeviceWriter('SampleRate',fs);
% 
% pHold = ones(hopSize,1);
% 
% speechProbLog = []; 
% 
% while ~isDone(afr)
%     x = afr();
%     n = write(inputBuffer,x);
% 
%     overlappedInput = read(inputBuffer,frameSize,overlapSize);
% 
%     p = VAD(overlappedInput);
% 
%     pHold(end) = p;
%     scope(x,pHold)
% 
%     player(x);
% 
%     pHold(:) = p;
% 
%     speechProbLog = [speechProbLog; p];
% end
% 
% t = (0:length(speechProbLog)-1) * hopSize/fs;
% % plot(t, speechProbLog), xlabel('Time (s)'), ylabel('Speech Probability')
% 
% count = 0;
% flag = 0;
% indexer = [];
% 
% for k = 1:length(speechProbLog)
% 
%     if speechProbLog(k) >= 0.95
%         count = count+1;
% 
%     end
% 
%     if count == 35
%         indexer = [indexer,0];
%         indexer(end - 35 + 1 : end) = 1; %This should replace instead of add
%         flag = 1;
%         count = 0;
%     elseif flag == 1;
%         indexer = [indexer,1];
% 
%         if speechProbLog(k) < 0.95
%             flag = 0;
%         end
% 
% 
%     else 
%         indexer = [indexer,0];
% 
% 
% 
%     end
% 
% end
% 
% %%
% 
% idx_nor = 10 .* indexer;
% ix = 4.6432e-04:4.6432e-04:3;
% 
% op_idx = 10 - idx_nor;
% op_idx = abs(op_idx);
% 
% figure;
% 
% %plot(ix,idx_nor)
% area(ix,idx_nor,'FaceColor',[0.9882 0.7804 0.7804],'EdgeColor',[0.9882 0.7804 0.7804]); hold on
% area(ix,op_idx,'FaceColor',[0.7804 0.8392 0.9882],'EdgeColor',[0.7804 0.8392 0.9882]); hold off
% 
% ylim([-0.5 10])


%% plotting rasters & hists

% figure;                                         %Plot the target stimuli
% subplot(2,1,1)
% plot(target1)                                   %Stim  1
% title('Target 1')
% subplot(2,1,2)
% plot(target2)                                   %Stim  2
% title('Target 2')
% 
% sgtitle('Target Stimuli')

labels = {'-90°','0°','45°','90°'};
total_spike_data = [];

% figure;

for i = 1:15    %16:27 next subject: 151L 230206 ks2.5

    tunetype = all_data(i).tuning_type;

    for j = 1:length(labels)

        total_spike_data = [];

        figure;

        tlen = length(target1);
        xfac = 3/tlen;
        tx = xfac:xfac:3;

        at = abs(target1);
        mt = max(at);
        n_t = 10/mt;
        n_target1 = n_t .* at;
        % c_target1 = 5 + n_target1;

        plot(tx,n_target1,'Color',[0.7804 0.8392 0.9882 0.8])

        hold on
        % subplot(3,1,[1 2])

        % area(ix,idx_nor,'FaceColor',[0.9882 0.7804 0.7804],'EdgeColor',[0.9882 0.7804 0.7804]); hold on
        % area(ix,op_idx,'FaceColor',[0.7804 0.8392 0.9882],'EdgeColor',[0.7804 0.8392 0.9882]);

        for k = 1:10

            SpikeTimes = all_data(i).ctrl_tar1_timestamps{k,j};  %(10 x 4) %As it stands (look at animal at index 124 and the timestamps associated with the first location and iterate over all 10 trials)

            % times = find(reference_cell{j} == 1);

            b = transpose(repmat(SpikeTimes,1,3));                 %Reshape SpikeTimes
            y_lines = nan(1,length(SpikeTimes));                   %Create a vertical line at each spike time
            y_lines(1,:) = k-1;
            y_lines(2,:) = k;
            y_lines(3,:) = nan;
            h2 = plot(b,y_lines,'Color',[0 0 0 0.8],'LineWidth',0.4);   %Plot
            total_spike_data = [total_spike_data; SpikeTimes];      %Save data for PSTH plotting

        end

        counts = histcounts(total_spike_data,'BinWidth',0.02); %Bin width of 20ms
        domain = linspace(-1,4,length(counts));

        cn_fac = 10 / max(counts);
        counts_norm = counts .* cn_fac;

        plot(domain,counts_norm,'r','LineWidth',1.5)
        % area(ix,idx_nor,'FaceColor',[0.9882 0.7804 0.7804],'EdgeColor',[0.9882 0.7804 0.7804])

        % hold off

        firerate = length(total_spike_data)/5;
        % firerate = floor(firerate);

        %Pad the first second at the start (Pre-onset)
        pre_onset = zeros(195312,1);

        %Pad the offset until recording offset at t=4
        post_stim = zeros(195312+3360,1); %Add a second plus a little bit since the stim is not exactly 3 seconds according to sampling rate.

        %Combine everything
        target1_full = [pre_onset;target1;post_stim];

        % subplot(3,1,3)
        % plot(target1_full)         %Plot full target
        % xticks(0:length(target1_full)/10:length(target1_full))   %Align timing from sample space to real time.
        % xticklabels(-1:0.5:4)
        % xlim([0 length(target1_full)])


        hold off

        % xlabel('Target 1')
        %
        angle = string(labels(1,j));
        cell = i;
        tunty = string(tunetype);
        my_title = sprintf("Subject: 150R 230201 ks2.5 // Cell: %d // Angle: %s // Tuning Type: %s // Audio: Target1 /// Firing Rate: %g", cell, angle, tunty, firerate);
        title(my_title)
      
    end
end

% %Pad the first second at the start (Pre-onset)
% pre_onset = zeros(195312,1);
% 
% %Pad the offset until recording offset at t=4
% post_stim = zeros(195312+3360,1); %Add a second plus a little bit since the stim is not exactly 3 seconds according to sampling rate.
% 
% %Combine everything  
% target1_full = [pre_onset;target1;post_stim];
% 
% subplot(3,1,3)
% plot(target1_full)         %Plot full target
% xticks(0:length(target1_full)/10:length(target1_full))   %Align timing from sample space to real time.
% xticklabels(-1:0.5:4)
% xlim([0 length(target1_full)])
% 
% xlabel('Target 1')
% sgtitle('Subject: 616283 Angle: -90 Audio: Target1')
% 
% 