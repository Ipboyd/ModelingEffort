close all

profile on;

fig_num = 6;

% counter = 0;
counter = 25;


% implement check to first navigate to correct subfolder
curr_folder = pwd;
plot_folder = 'Plot_Structure_Graphs';


if ~endsWith(curr_folder, plot_folder)
    if ~isfolder(plot_folder)
        mkdir(plot_folder);
    end
    cd 'Plot_Structure_Graphs';
end


titles = {'On','Off','ROn','ROff','SOnOff','TD','X','C'};

channels = [1,2,3,4];

tic;
parfor j = 1:length(titles)
    node_name = titles{j};
    plotting_data = [];
    for m = subz
        location = data(m).config;
        % counter = counter + 1;
        for data_channel = channels
            for k = 1:10
                if k > 10
                    d = 10;
                else
                    d= 0;
                end
    
                %% Rasters
                %Grab the spikes that we are interested in
                On = transpose(data(m).spks.On.channel1(k,:));
                Off = transpose(data(m).spks.Off.channel1(k,:));
                Ron = transpose(data(m).spks.ROn.channel1(k,:));
                Roff = transpose(data(m).spks.ROff.channel1(k,:));
                SOnOff = transpose(data(m).spks.SOnOff.channel1(k,:));
                TD = transpose(data(m).spks.TD.channel1(k,:));
                X = transpose(data(m).spks.X.channel1(k,:));
                C = transpose(data(m).spks.C.channel1(k,:));
            
                reference_cell = {On,Off,Ron,Roff,SOnOff,TD,X,C};
                
                times = find(reference_cell{j} == 1);
                b = transpose(repmat(times,1,3));
                y_lines = nan(1,length(times));
                y_lines(1,:) = k-1-d;
                y_lines(2,:) = k-d;
                y_lines(3,:) = nan;
                figure('Visible', 'off');
                plotting_data = [plotting_data;times];
    
                % Optimize figure properties
                % set(gcf, 'Toolbar', 'none', 'Menubar', 'none');
    
                plot(b,y_lines,'Color',[0 0 0 0.2],'LineWidth',0.3); hold on;
    
                            
            end
            channel_string = join(["Channel", data_channel], ' ');
            graph_title = [node_name, channel_string, location];
            title(graph_title, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
    
            RASTER_file_name = [node_name num2str(data_channel) '_' location '_RASTER' '.fig'];
            saveas(gcf, RASTER_file_name);
            close(gcf);

            raw_freq_data = histcounts(plotting_data, BinWidth=200);
            avg_data1 = movmean(raw_freq_data,3);
        
            %Conver to frequency from spike count
            % avg_data = (avg_data1*50/10)*10/150*75/40;
            
            figure('Visible', 'off');
            % Optimize figure properties
            % set(gcf, 'Toolbar', 'none', 'Menubar', 'none');
            
            plot(avg_data1,'r',LineWidth=0.5);
        
            channel_string = join(["Channel", data_channel], ' ');
            graph_title = [node_name, channel_string, location];
            title(graph_title, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial');
        
            PSTH_file_name = [node_name num2str(data_channel) '_' location '_PSTH' '.fig'];
            saveas(gcf, PSTH_file_name); % Save as MATLAB figure file
            close(gcf);
        end

    % if contains(titles{j}, 'S')
    %     plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'r',LineWidth=0.5);
    % else
    %     plot(linspace(0,ending_sample-starting_sample,length(avg_data)),avg_data,'k',LineWidth=0.5);
    % end
    % title(titles{j})
    % 
    % %Fig 4/5
    % xlim([3500 13500])
    % 
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
    %         %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','ZoomedVersion','.svg']) % svg
    %         figure(91)
    %         %Range is changeable (Just looking for a relavent pattern here)
    %         plot(holder2(9000:15000),'r')
    %         ylim([-80 0])
    %         %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','PVspikes','.svg']) % svg
    % 
    %     elseif contains(titles{j}, 'R2on')
    %         figure(90)
    %         plot(linspace(0,ending_sample-starting_sample,length(avg_data1)),avg_data1,'k',LineWidth=0.5); hold on
    %         figure(92)
    %         plot(holder(9000:15000),'k')
    %         ylim([-80 0])
    %         %print(gcf,'-vector','-dsvg',['C:\Users\ipboy\Desktop\Modeling Paper\Figures\Figure',num2str(fig_num),'\FpUpdate\','Espikes','.svg']) % svg
    %     end
    % 
    % 
    % end
    end

end
toc;

cd ..

profile viewer;