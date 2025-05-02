close all
%Set how long you want the recording to be
starting_sample = 0;
ending_sample = 35000;

data_channel = 1;
fig_num = 8;

titles = {'R2on','R2OnLaser'};

for j = 1:2
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
        R2on = transpose(data(subz).spks.R2On(data_channel).channel1(k,:));
        R2onLaser = transpose(data(subz).spks.R2On(data_channel+1).channel1(k,:));
        %R1off = transpose(data(subz).spks.R1Off(data_channel).channel1(k,:));
        %R2off = transpose(data(subz).spks.R2Off(data_channel).channel1(k,:));
        %S1OnOff = transpose(data(subz).spks.S1OnOff(data_channel).channel1(k,:));
        %S2OnOff = transpose(data(subz).spks.S2OnOff(data_channel).channel1(k,:));


        reference_cell = {R2on,R2onLaser};
        
        times = find(reference_cell{j} == 1);
        b = transpose(repmat(times,1,3));
        y_lines = nan(1,length(times));
        y_lines(1,:) = k-1-d;
        y_lines(2,:) = k-d;
        y_lines(3,:) = nan;
        figure(25+j)
        plotting_data = [plotting_data;times];
        h2 = plot(b,y_lines,'Color',[0 0 0],'LineWidth',0.3); hold on

    end

    
    raw_freq_data = histcounts(plotting_data, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
    avg_data1 = movmean(raw_freq_data,3);
    %avg_data = (avg_data1*50/10)*10/150;
    %avg_data = (avg_data1*50/10)*10/150*75/40;
    %if contains(titles{j}, 'S')
    %    plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'r',LineWidth=0.5);
    %else
    %    plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'k',LineWidth=0.5);
    %end
    title(titles{j})
    
    %
    xlim([1 35000])

    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'.svg']) % svg

    %xlim([8500 20500])
    %xlim([20000 29000])
    %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'Zoomed','.svg']) % svg
    
    %figure(30+j)
    %plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'k',LineWidth=0.5);
    %xlim([8500 20500])
    %xlim([20000 29000])
    %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'ZoomedPSTH','.svg']) % svg

    figure(35+j);
    nbins = 175;
    histogram(plotting_data,nbins,'FaceColor','k', 'FaceAlpha', 1);
   
    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'Histogram','.svg']) % svg


    plotting_data = [];
    for k = 11:20
        if k > 10
            d = 10;
        else
            d= 0;
        end

        %% Rasters
        R2on = transpose(data(subz).spks.R2On(data_channel).channel1(k,:));
        R2onLaser = transpose(data(subz).spks.R2On(data_channel+1).channel1(k,:));

        reference_cell = {R2on,R2onLaser};
        
        times = find(reference_cell{j} == 1);
        b = transpose(repmat(times,1,3));
        y_lines = nan(1,length(times));
        y_lines(1,:) = k-1-d;
        y_lines(2,:) = k-d;
        y_lines(3,:) = nan;
        figure(27+j)
        plotting_data = [plotting_data;times];
        h2 = plot(b,y_lines,'Color',[0 0 0],'LineWidth',0.3); hold on

    end

    
    raw_freq_data = histcounts(plotting_data, BinWidth=200,BinLimits=[0 ending_sample-starting_sample]);
    avg_data = movmean(raw_freq_data,3);
    %avg_data = (avg_data1*50/10)*10/150;
    %avg_data = (avg_data1*50/10)*10/150*75/40;
    title(titles{j})
    
    %
    xlim([1 35000])

    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'S2','.svg']) % svg
    
    %xlim([8500 20500])
    xlim([27000 30000])
    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'Zoomed','.svg']) % svg
    
    figure(30+j)
    plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'k',LineWidth=0.5);
    %xlim([8500 20500])
    xlim([27000 30000])
    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'ZoomedPSTH','.svg']) % svg

    figure(37+j);
    nbins = 175;
    histogram(plotting_data,nbins,'FaceColor','k', 'FaceAlpha', 1);
   
    print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\',titles{j},'HistogramS2','.svg']) % svg

    %%Plot zoom raster & Voltage plot (Fig 5)
    if fig_num == 5 || fig_num == 6
        
        indexs = find(snn_out(data_channel).R2On_V_spikes == 1);
        indexs2 = find(snn_out(data_channel).S2OnOff_V_spikes == 1);
        holder = snn_out(data_channel).R2On_V;
        holder(indexs) = 0;
        holder2 = snn_out(data_channel).S2OnOff_V;
        holder2(indexs2) = 0;
        
        
        if contains(titles{j}, 'S2OnOff')
            figure(90)
            plot(linspace(0,ending_sample-starting_sample,length(avg_data1)),avg_data1,'r',LineWidth=0.5); hold on
            %xlim([8500 13500])
            xlim([5500 10500])
            print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','ZoomedVersion','.svg']) % svg
            figure(91)
            %Range is changeable (Just looking for a relavent pattern here)
            plot(holder2(9000:15000),'r')
            ylim([-80 0])
            print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','PVspikes','.svg']) % svg
            
        elseif contains(titles{j}, 'R2on')
            figure(90)
            plot(linspace(0,ending_sample-starting_sample,length(avg_data1)),avg_data1,'k',LineWidth=0.5); hold on
            figure(92)
            plot(holder(9000:15000),'k')
            ylim([-80 0])
            print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','Espikes','.svg']) % svg
        end

        
    end

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
