%Pad the end for now. It looks like the way that we set things up
%everything is off by 1 index?

%stored_val = single(x);
%stored_val(35000) = 0;
%reshape(stored_val)

% strored_val_reshaped = reshape(xs, [349990/10 ,10]);
% 
% SpikeTimes_all = cell(1, size(strored_val_reshaped, 2));  % 1x20 cell array
% 
% % Loop over each neuron (column)
% for n = 1:size(strored_val_reshaped, 2)
%     % Find spike times (rows where value is 1)
%     SpikeTimes_all{n} = (find(strored_val_reshaped(:, n))/10000)-0.3;
% end
% 
% cd(userpath);
% cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')
% 
% load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
% load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz
% 
% figure;                                         %Plot the target stimuli
% subplot(2,1,1)
% plot(target1)                                   %Stim  1
% subplot(2,1,2)
% plot(target2)                                   %Stim  2
% 
% total_spike_data = [];
% 
% figure;
% 
% subplot(3,1,[1 2])
% 
% for k = 1:10
% 
%     SpikeTimes = SpikeTimes_all{k};%all_data(124).ctrl_tar1_timestamps{k,1};  %(10 x 4) %As it stands (look at animal at index 124 and the timestamps associated with the first location and iterate over all 10 trials)
%     b = transpose(repmat(SpikeTimes,1,3));                 %Reshape SpikeTimes
%     y_lines = nan(1,length(SpikeTimes));                   %Create a vertical line at each spike time
%     y_lines(1,:) = k-1;
%     y_lines(2,:) = k;
%     y_lines(3,:) = nan;
%     h2 = plot(b,y_lines,'Color',[0 0 0 0.8],'LineWidth',0.4); hold on   %Plot
%     total_spike_data = [total_spike_data; SpikeTimes];      %Save data for PSTH plotting
% 
% end
% 
% counts = histcounts(total_spike_data,'BinWidth',0.02); %Bin width of 20ms
% domain = linspace(0.1,3,length(counts));
% 
% plot(domain,counts,'r','LineWidth',1.5)
% 
% xlim([-1 4])
% 
% hold off
% 
% %Pad the first second at the start (Pre-onset)
% pre_onset = zeros(195312,1);
% 
% %Pad the offset until recording offset at t=4
% post_stim = zeros(195312+3360,1); %Add a second plus a little bit since the stim is not exactly 3 seconds accroding to sampling rate.
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
% %sgtitle('Subject: 616283 Angle: -90 Audio: Target1')

spikes_spks = x{1};
spikes_epoch = spikes_spks{1};

n_trials = length(spikes_epoch);
n_outputs = 1;  % On_V, Off_V, R1On_V, ..., S2OnOff_V

spikes = zeros(n_trials, n_outputs);
spike_times = cell(n_trials, n_outputs);  % Time indices where spikes occurred



for i = 1:n_trials
    %trial_outputs = single(spikes_epoch{i});
    for j = 1:n_outputs
        %data = double(trial_outputs{j});       % [34999 × 1]
        spikes(i, j) = single(spikes_epoch{i});
        %spike_times{i, j} = find(data);        % Get indices where spike == 1
    end
end


spikes = reshape(spikes,34998,11);
%spike_times = mod(find(spikes),34998);

spike_times = {};

for a = 1:10
    spike_times{end+1} = find(spikes(:,a+1));
end

%spike_times{i, j} = find(matlab_holder_reshaped); 

%return [On_V_spikes,Off_V_spikes,R1On_V_spikes,R1Off_V_spikes,R2On_V_spikes,S1OnOff_V_spikes,S2OnOff_V_spikes]\n\n'''



%Set how long you want the recording to be
starting_sample = 0;
ending_sample = 35000;

data_channel = 1;
fig_num = 6;


%titles = {'on','off','R1on','R1off','R2On','S1OnOff','S2OnOff'};

titles = {'on'};

for j = 1:size(spike_times,1)
    
    plotting_data = [];
    for k = 1:10
        if k > 10
            d = 10;
        else
            d= 0;
        end

        %% Rasters
        %Grab the spikes that we are interested in
        %on = transpose(data(subz).spks.On(data_channel).channel1(k,:));
        %off = transpose(data(subz).spks.Off(data_channel).channel1(k,:));
        %R1on = transpose(data(subz).spks.R1On(data_channel).channel1(k,:));
        %R2on = transpose(data(subz).spks.R2On(data_channel).channel1(k,:));
        %R1off = transpose(data(subz).spks.R1Off(data_channel).channel1(k,:));
        %R2off = transpose(data(subz).spks.R2Off(data_channel).channel1(k,:));
        %S1OnOff = transpose(data(subz).spks.S1OnOff(data_channel).channel1(k,:));
        %S2OnOff = transpose(data(subz).spks.S2OnOff(data_channel).channel1(k,:));

        %reference_cell = {on,off,R1on,R2on,R1off,R2off,S1OnOff,S2OnOff};
        
        
        

        times = spike_times{j,k};
        b = transpose(repmat(times,1,3));
        y_lines = nan(1,length(times));
        y_lines(1,:) = k-1-d;
        y_lines(2,:) = k-d;
        y_lines(3,:) = nan;
        figure(25+j)
        plotting_data = [plotting_data;times];
        h2 = plot(b,y_lines,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on

    end

    
    raw_freq_data = histcounts(plotting_data, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
    avg_data1 = movmean(raw_freq_data,3);
    %avg_data = (avg_data1*50/10)*10/150;
    avg_data = (avg_data1*50/10)*10/150*75/40;
    if contains(titles{j}, 'S')
        plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'r',LineWidth=0.5);
    else
        plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'k',LineWidth=0.5);
    end
    title(titles{j})
    
    %Fig 4/5
    xlim([1 35000])

    % print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'.svg']) % svg
    % 
    % 
    % %%Plot zoom raster & Voltage plot (Fig 5)
    % if fig_num == 5 || fig_num == 6
    % 
    %     indexs = find(snn_out(data_channel).R2On_V_spikes == 1);
    %     indexs2 = find(snn_out(data_channel).S2OnOff_V_spikes == 1);
    %     holder = snn_out(data_channel).R2On_V;
    %     holder(indexs) = 0;
    %     holder2 = snn_out(data_channel).S2OnOff_V;
    %     holder2(indexs2) = 0;
    % 
    % 
    %     if contains(titles{j}, 'S2OnOff')
    %         figure(90)
    %         plot(linspace(0,ending_sample-starting_sample,length(avg_data1)),avg_data1,'r',LineWidth=0.5); hold on
    %         %xlim([8500 13500])
    %         xlim([5500 10500])
    %         print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','ZoomedVersion','.svg']) % svg
    %         figure(91)
    %         %Range is changeable (Just looking for a relavent pattern here)
    %         plot(holder2(9000:15000),'r')
    %         ylim([-80 0])
    %         print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','PVspikes','.svg']) % svg
    % 
    %     elseif contains(titles{j}, 'R2on')
    %         figure(90)
    %         plot(linspace(0,ending_sample-starting_sample,length(avg_data1)),avg_data1,'k',LineWidth=0.5); hold on
    %         figure(92)
    %         plot(holder(9000:15000),'k')
    %         ylim([-80 0])
    %         print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','Espikes','.svg']) % svg
    %     end
    % 
    % 
    % end

end





% 
% 
% 
% 
%     end
% 
%     if k == 10
%         %ToDo insert the PSTH plot for each raster.
%     end
% 
% 
%     %a = transpose(data(subz).spks.R2Off(2).channel1(k,:));
%     %a2 = transpose(data(subz).spks.R1On(1).channel1(k,:));
%     %a = a(starting_sample:ending_sample);
%     %a2 = a2(starting_sample:ending_sample);
%     %times = find(a == 1);
%     %times2 = find(a2 == 1);
%     %b = transpose(repmat(times,1,3));
%     % b2 = transpose(repmat(times2,1,3));
%     % 
%     % if k > 10
%     %     d = 10;
%     % else
%     %     d= 0;
%     % end
% 
%     %plotting_data(k,:) = a;
%     %plotting_data2 = [plotting_data2;times];
%     %plotting_data3 = [plotting_data3;times2];
% 
%     %Start with prealocated size using nan
%     % y_lines = nan(1,length(times));
%     % y_lines(1,:) = k-1-d;
%     % y_lines(2,:) = k-d;
%     % y_lines(3,:) = nan;
%     % 
%     % y_lines2 = nan(1,length(times2));
%     % y_lines2(1,:) = k-1-d;
%     % y_lines2(2,:) = k-d;
%     % y_lines2(3,:) = nan;
%     % 
%     % 
%     % %Plot the rasters
%     % figure(25)
%     % h2 = plot(b,y_lines,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on
%     % print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure8\8HzRasterZoom','.svg']) % svg
%     % 
%     % xlim([1 35000])
%     %xlim([8500 20500])
%     %xlim([27000 30000])
%     %figure(13)
%     %h4 = plot(b2,y_lines2,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on
%     %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure8\30HzRasterZoom','.svg']) % svg
%     %xlim([8500 20500])
%     %xlim([1 35000])
%     %xlim([27000 30000])
% 
% end
% 
% 
% %Adjust the size of the figure with figure handler (h)
% h.Position = [100 100 850 400];
% 
% %% PSTH
% %State if we are doing figure b or d
% b = true;
% %30 = 150hz
% 
% raw_freq_data = histcounts(plotting_data2, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
% raw_freq_data2 = histcounts(plotting_data3, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
% 
% avg_data = movmean(raw_freq_data,3);
% avg_data2 = movmean(raw_freq_data2,3);
% 
% figure(36);
% nbins = 175;
% histogram(plotting_data2,nbins,'FaceColor','k', 'FaceAlpha', 1);
% 
% figure(37);
% nbins = 175;
% histogram(plotting_data3,nbins,'FaceColor','k', 'FaceAlpha', 1);
% 
% % h = figure(25);
% % %hist_data = sum(plotting_data);
% % 
% % %histogram(plotting_data2, BinWidth=200)
% % if b == true
% %     %avg_data
% % 
% %     disp('h')
% %     %signal = movmean(sum(plotting_data),200);
% %     %sig_max = max(signal);
% %     %Scale
% %     avg_data = (avg_data*50/10)*10/150;
% %     %avg_data2 = (avg_data2*50/10)*10/150;
% % 
% % 
% % 
% % 
% %     %Create matching x-domain for PSTH
% %     xdom = linspace(0,ending_sample-starting_sample,length(avg_data));
% % 
% %     %plot psth
% %     %figure(99);
% %     plot(xdom,avg_data,'k',LineWidth=0.5); hold on
% %     %plot(xdom,avg_data2,'r',LineWidth=0.5);
% %     %xlim([8500 20500])
% %     xlim([3500 13500])
% %     %xlim([27000 30000])
% % else
% %     figure(2);
% %     plot(avg_data,'k',LineWidth=0.5)
% % end

