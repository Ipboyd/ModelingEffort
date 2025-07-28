%load(['/Users/lbowman/Desktop/Research/all_units_info_with_polished_criteria_modified_perf.mat'],'all_data');
%load(['/Users/lbowman/Desktop/Research/sound_files.mat'],'sampleRate','target1','target2');
%load(['/Users/ipboy/Downloads/all_units_info_with_polished_criteria_modified_perf.mat'],'all_data');
%load(['/Users/ipboy/Downloads/sound_files.mat'],'sampleRate','target1','target2');  

cd(userpath);
cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')


load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

figure;                                         %Plot the target stimuli
subplot(2,1,1)
plot(target1)                                   %Stim  1
subplot(2,1,2)
plot(target2)                                   %Stim  2

total_spike_data = [];

figure;

subplot(3,1,[1 2])

for k = 1:10
    
    SpikeTimes = all_data(7).ctrl_tar1_timestamps{k,1};  %(10 x 4) %As it stands (look at animal at index 124 and the timestamps associated with the first location and iterate over all 10 trials)
    b = transpose(repmat(SpikeTimes,1,3));                 %Reshape SpikeTimes
    y_lines = nan(1,length(SpikeTimes));                   %Create a vertical line at each spike time
    y_lines(1,:) = k-1;
    y_lines(2,:) = k;
    y_lines(3,:) = nan;
    h2 = plot(b,y_lines,'Color',[0 0 0 0.8],'LineWidth',0.4); hold on   %Plot
    total_spike_data = [total_spike_data; SpikeTimes];      %Save data for PSTH plotting

end



counts = histcounts(total_spike_data,'BinWidth',0.02); %Bin width of 20ms
domain = linspace(-1,4,length(counts));

%plot(domain,counts,'r','LineWidth',1.5)

hold off

%Pad the first second at the start (Pre-onset)
pre_onset = zeros(195312,1);

%Pad the offset until recording offset at t=4
post_stim = zeros(195312+3360,1); %Add a second plus a little bit since the stim is not exactly 3 seconds accroding to sampling rate.

%Combine everything  
target1_full = [pre_onset;target1;post_stim];

subplot(3,1,3)
plot(target1_full)         %Plot full target
xticks(0:length(target1_full)/10:length(target1_full))   %Align timing from sample space to real time.
xticklabels(-1:0.5:4)
xlim([0 length(target1_full)])

xlabel('Target 1')
sgtitle('Subject: 616283 Angle: -90 Audio: Target1')


