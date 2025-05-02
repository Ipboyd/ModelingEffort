%Target spike population
%target = 'R2On';
%h = figure(25);

%Set how long you want the recording to be
starting_sample = 0;
ending_sample = 35000;
xdom = linspace(0,ending_sample-starting_sample,length(avg_data));
%plotting_data = [];
plotting_data2 = [];
plotting_data3 = [];

data_channel = 1;

%18 3/4

%Go through all of the 
for k = 1:10

    if k > 10
        d = 10;
    else
        d= 0;
    end
    
    %% Rasters
    %Grab the spikes that we are interested in
    on = transpose(data(subz).spks.On(data_channel).channel1(k,:));
    off = transpose(data(subz).spks.Off(data_channel).channel1(k,:));
    R1on = transpose(data(subz).spks.R1On(data_channel).channel1(k,:));
    R2on = transpose(data(subz).spks.R2On(data_channel).channel1(k,:));
    R1off = transpose(data(subz).spks.R1Off(data_channel).channel1(k,:));
    R2off = transpose(data(subz).spks.R2Off(data_channel).channel1(k,:));
    S1OnOff = transpose(data(subz).spks.S1OnOff(data_channel).channel1(k,:));
    S2OnOff = transpose(data(subz).spks.S2OnOff(data_channel).channel1(k,:));
    
    reference_cell = {on,off,R1on,R2on,R1off,R2off,S1OnOff,S2OnOff};
    
    %Look for the spike times in all of the cells
    for j = 1:length(reference_cell)
        times = find(reference_cell{j} == 1);
        b = transpose(repmat(times,1,3));
        y_lines = nan(1,length(times));
        y_lines(1,:) = k-1-d;
        y_lines(2,:) = k-d;
        y_lines(3,:) = nan;
        figure(25+j)
        
        h2 = plot(b,y_lines,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on
        raw_freq_data = histcounts(plotting_data, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
        avg_data = movmean(raw_freq_data,3);
        plot(xdom,avg_data,'k',LineWidth=0.5); hold on
        
        %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure8\8HzRasterZoom','.svg']) % svg
        xlim([1 35000])
    end

    if k == 10
        %ToDo insert the PSTH plot for each raster.
    end


    %a = transpose(data(subz).spks.R2Off(2).channel1(k,:));
    %a2 = transpose(data(subz).spks.R1On(1).channel1(k,:));
    %a = a(starting_sample:ending_sample);
    %a2 = a2(starting_sample:ending_sample);
    %times = find(a == 1);
    %times2 = find(a2 == 1);
    %b = transpose(repmat(times,1,3));
    % b2 = transpose(repmat(times2,1,3));
    % 
    % if k > 10
    %     d = 10;
    % else
    %     d= 0;
    % end

    %plotting_data(k,:) = a;
    %plotting_data2 = [plotting_data2;times];
    %plotting_data3 = [plotting_data3;times2];

    %Start with prealocated size using nan
    % y_lines = nan(1,length(times));
    % y_lines(1,:) = k-1-d;
    % y_lines(2,:) = k-d;
    % y_lines(3,:) = nan;
    % 
    % y_lines2 = nan(1,length(times2));
    % y_lines2(1,:) = k-1-d;
    % y_lines2(2,:) = k-d;
    % y_lines2(3,:) = nan;
    % 
    % 
    % %Plot the rasters
    % figure(25)
    % h2 = plot(b,y_lines,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on
    % print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure8\8HzRasterZoom','.svg']) % svg
    % 
    % xlim([1 35000])
    %xlim([8500 20500])
    %xlim([27000 30000])
    %figure(13)
    %h4 = plot(b2,y_lines2,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on
    %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure8\30HzRasterZoom','.svg']) % svg
    %xlim([8500 20500])
    %xlim([1 35000])
    %xlim([27000 30000])

end


%Adjust the size of the figure with figure handler (h)
h.Position = [100 100 850 400];

%% PSTH
%State if we are doing figure b or d
b = true;
%30 = 150hz

raw_freq_data = histcounts(plotting_data2, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
raw_freq_data2 = histcounts(plotting_data3, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);

avg_data = movmean(raw_freq_data,3);
avg_data2 = movmean(raw_freq_data2,3);

figure(36);
nbins = 175;
histogram(plotting_data2,nbins,'FaceColor','k', 'FaceAlpha', 1);

figure(37);
nbins = 175;
histogram(plotting_data3,nbins,'FaceColor','k', 'FaceAlpha', 1);

% h = figure(25);
% %hist_data = sum(plotting_data);
% 
% %histogram(plotting_data2, BinWidth=200)
% if b == true
%     %avg_data
% 
%     disp('h')
%     %signal = movmean(sum(plotting_data),200);
%     %sig_max = max(signal);
%     %Scale
%     avg_data = (avg_data*50/10)*10/150;
%     %avg_data2 = (avg_data2*50/10)*10/150;
% 
% 
% 
% 
%     %Create matching x-domain for PSTH
%     xdom = linspace(0,ending_sample-starting_sample,length(avg_data));
% 
%     %plot psth
%     %figure(99);
%     plot(xdom,avg_data,'k',LineWidth=0.5); hold on
%     %plot(xdom,avg_data2,'r',LineWidth=0.5);
%     %xlim([8500 20500])
%     xlim([3500 13500])
%     %xlim([27000 30000])
% else
%     figure(2);
%     plot(avg_data,'k',LineWidth=0.5)
% end
