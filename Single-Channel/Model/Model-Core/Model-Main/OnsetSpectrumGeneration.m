clear all

addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting\PicturesToFit')

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');

folder = 'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells_both';
files  = dir(fullfile(folder,'*.mat'));
[~,idx] = sort([files.datenum]);  % optional: sort by time
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

%The following fixes this. Comment out if you do another new run.
idx = idx - 1; %Shift everything one over
idx(1) = 122; %Make the 122nd the first.
idx(123:220) = 123:220;
files = files(idx);

label_struct = load('x_set_labels.mat');
cmap  = [ 57 106 177;   % Both  (blue-ish)
          62 150  81;   % Onset    (green-ish)
         204  37  41;   % Offset  (red-ish)
         160 160 160 ] / 255;  % Neither  (gray)
%%

%Extract parameters
all_params = [];
colors = [];
for k = 1:182
    fpath = fullfile(files(k).folder, files(k).name);
    data  = load(fpath);
    
    %Find min loss
    losses = squeeze(data.losses(:,2,:));
    min_losses = min(min(losses));
    idx_val = find(losses==min_losses);
    
    %Dont allow for ties
    sizes = size(losses);
    index2 = ceil(idx_val(1)/sizes(1));
    index1 = idx_val(1) - ((index2-1)*sizes(1));
    all_params = [all_params;data.param_tracker(index1,:,index2)];

    cur_cell = label_struct.our_struct(k).n90;
    colors = [colors;cmap(find([cur_cell.is_Both,cur_cell.is_Onset,cur_cell.is_Offset,cur_cell.is_Neither]),:)];
end


%%%%%%%%%%%%%%%%%%% LOOKING AT ONSETS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
close all
eps = 0.001;

%Create a spectrum of relative "onsetness"
rel_onsetness = all_params(:,1)./(all_params(:,2)+eps);

%Create a specturm of absolute "onsetness"
abs_onsetness = all_params(:,1);

%Sort the arrays
[rel_sort,rel_sort_idx] = sort(rel_onsetness);
[abs_sort,abs_sort_idx] = sort(abs_onsetness);

colors_sorted_rel = colors(rel_sort_idx,:);
colors_sorted_abs = colors(abs_sort_idx,:);

%Plot the spectrum
figure;
subplot(2,1,1)
%histogram(rel_sort(find(colors_sorted_rel(:,1) == cmap(1,1))),20,"FaceColor",cmap(1,:)); hold on
histogram(rel_sort(find(colors_sorted_rel(:,1) == cmap(2,1))),"BinEdges",[0:3:60],"FaceColor",cmap(2,:)); hold on
histogram(rel_sort(find(colors_sorted_rel(:,1) == cmap(3,1))),"BinEdges",[0:3:60],"FaceColor",cmap(3,:)); hold on
%histogram(rel_sort(find(colors_sorted_rel(:,1) == cmap(4,1))),20,"FaceColor",cmap(4,:)); hold on
xlim([0,60])

subplot(2,1,2)
scatter(rel_sort,zeros(1,length(rel_sort)),72,colors_sorted_rel,"filled","o","MarkerFaceAlpha",0.5)
for i = 1:length(rel_sort)
    text(rel_sort(i), 0, sprintf('%d', rel_sort_idx(i)), ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right', ...
        'FontSize', 8, ...
        'Color', 'k');
end

sgtitle('Relative "Onsetness"')

[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(149,files);

%Look at the sim Rasters
figure(12);
subplot(5,1,1)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(149) + '     --    Weight: ' + rel_onsetness(149))

figure(13);
subplot(5,1,1)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(149) + '     --    Weight: ' + rel_onsetness(149))

[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(152,files);

figure(12);
subplot(5,1,2)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(152) + '     --    Weight: ' + rel_onsetness(152))

figure(13);
subplot(5,1,2)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(152) + '     --    Weight: ' + rel_onsetness(152))

[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(63,files);

figure(12);
subplot(5,1,3)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(63)+ '     --    Weight: ' + rel_onsetness(63))

figure(13);
subplot(5,1,3)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(63) + '     --    Weight: ' + rel_onsetness(63))


[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(133,files);

figure(12);
subplot(5,1,4)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(133) + '     --    Weight: ' + rel_onsetness(133))

figure(13);
subplot(5,1,4)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(133) + '     --    Weight: ' + rel_onsetness(133))

figure(12);
subplot(5,1,5)
plot(target1)
xlim([0, length(target1)])
xticklabels('')
yticklabels('')
sgtitle('Relative "Onsetness"')

figure(13);
subplot(5,1,5)
plot(target1)
xlim([0, length(target1)])
xticklabels('')
yticklabels('')
sgtitle('Relative "Onsetness"')

figure;
subplot(2,1,1)
%histogram(abs_sort(find(colors_sorted_abs(:,1) == cmap(1,1))),20,"FaceColor",cmap(1,:)); hold on
histogram(abs_sort(find(colors_sorted_abs(:,1) == cmap(2,1))),20,"FaceColor",cmap(2,:)); hold on
histogram(abs_sort(find(colors_sorted_abs(:,1) == cmap(3,1))),20,"FaceColor",cmap(3,:)); hold on
%histogram(abs_sort(find(colors_sorted_abs(:,1) == cmap(4,1))),20,"FaceColor",cmap(4,:)); hold on
xlim([0,0.09])

subplot(2,1,2)
scatter(abs_sort,zeros(1,length(abs_sort)),72,colors_sorted_abs,"filled","o","MarkerFaceAlpha",0.5)

for i = 1:length(abs_sort)
    text(abs_sort(i), 0, sprintf('%d', abs_sort_idx(i)), ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right', ...
        'FontSize', 8, ...
        'Color', 'k');
end

sgtitle('Absolute "Onsetness"')


[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(63,files);

%Look at the sim Rasters
figure(10);
subplot(5,1,1)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(63) + '     --    Weight: ' + abs_onsetness(63))

figure(11);
subplot(5,1,1)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(63) + '     --    Weight: ' + abs_onsetness(63))

[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(13,files);

figure(10);
subplot(5,1,2)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(13) + '     --    Weight: ' + abs_onsetness(13))

figure(11);
subplot(5,1,2)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(13) + '     --    Weight: ' + abs_onsetness(13))

[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(48,files);

figure(10);
subplot(5,1,3)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(48)+ '     --    Weight: ' + abs_onsetness(48))

figure(11);
subplot(5,1,3)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(48) + '     --    Weight: ' + abs_onsetness(48))


[sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(109,files);

figure(10);
subplot(5,1,4)
plot(sim_entries,sim_ticks,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
title('Sim Raster: Cell ' + string(109) + '     --    Weight: ' + abs_onsetness(109))

figure(11);
subplot(5,1,4)
plot(sim_PSTH,'b',LineWidth=2)
title('Sim PSTH: Cell ' + string(109) + '     --    Weight: ' + abs_onsetness(109))

figure(10);
subplot(5,1,5)
plot(target1)
xlim([0, length(target1)])
xticklabels('')
yticklabels('')
sgtitle('Absolute "Onsetness"')

figure(11);
subplot(5,1,5)
plot(target1)
xlim([0, length(target1)])
xticklabels('')
yticklabels('')
sgtitle('Absolute "Onsetness"')




function [sim_entries,sim_ticks,data_entires,data_ticks,sim_PSTH,data_PSTH] = get_data(point,files)
    
    bin_width = 200; %In 0.1 ms
    fpath = fullfile(files(point).folder, files(point).name);
    data  = load(fpath);   
    sim = data.best_output;
    
    num_bins = floor(size(sim,2)/bin_width);
    remainder = mod(size(sim,2),bin_width);
    
    sim_trim = sim(:,remainder+1:end); 
    
    index_course = 1:size(sim_trim,2); 
    
    sim_scaled = sim_trim.*index_course;
    bin_edges = 1:bin_width:size(sim_scaled,2)+1;
    sim_PSTH = histcounts(sim_scaled,bin_edges);
    [trial_sim,indicy_sim] = find(sim_scaled);
    sim_entries = [indicy_sim,indicy_sim]';
    sim_ticks = [trial_sim,trial_sim-1]'; %Trial and Trial-1
    
    % plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
    % yticklabels('')
    % xticklabels('')
    % xlabel('Time')
    % ylabel('Trial')
    % title('Sim Raster: Cell ' + string(point1))
    
    %subplot(4,4,[7,8])

    data = load('picture_fit' + string(point) + 'contra.mat').picture;
    
    num_bins = floor(size(data,2)/bin_width);
    remainder = mod(size(data,2),bin_width);
    
    sim_trim = data(:,remainder+1:end); 
    
    index_course = 1:size(sim_trim,2); 
    
    sim_scaled = sim_trim.*index_course;
    bin_edges = 1:bin_width:size(sim_scaled,2)+1;
    data_PSTH = histcounts(sim_scaled,bin_edges);
    [trial_sim,indicy_sim] = find(sim_scaled);
    data_entires = [indicy_sim,indicy_sim]';
    data_ticks = [trial_sim,trial_sim-1]'; %Trial and Trial-1
    
    % [r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
    % [rmax, idx] = max(r);
    % 
    % plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
    % yticklabels('')
    % xticklabels('')
    % xlabel('Time')
    % ylabel('Trial')
    % title('Data Raster: Cell ' + string(point1) + ' Max Xcorr = ' + string(rmax))
    
   

end