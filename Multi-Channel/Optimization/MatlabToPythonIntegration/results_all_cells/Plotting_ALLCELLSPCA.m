clear all

addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting\PicturesToFit')

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');

folder = 'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells_both';
files  = dir(fullfile(folder,'*.mat'));
[~,idx] = sort([files.datenum]);  % optional: sort by time


%You forgot to run the first cell because of this things need to be
%rearanged

%The following fixes this. Comment out if you do another new run.
idx = idx - 1; %Shift everything one over
idx(1) = 122; %Make the 122nd the first.

files = files(idx);


label_struct = load('x_set_labels.mat');

layers = ["L2/3","L4","L5/6","NaN"];
cmap  = [ 57 106 177;   % L2/3  (blue-ish)
          62 150  81;   % L4    (green-ish)
         204  37  41;   % L5/6  (red-ish)
         160 160 160 ] / 255;  % NaN  (gray)

idx4 = strcmp({all_data.layer}, 'L4');
idx56 = strcmp({all_data.layer}, 'L5/6');
idx23 = strcmp({all_data.layer}, 'L2/3');
idxnan = strcmp({all_data.layer}, 'NaN');


l4_params = [];
l23_params = [];
l56_params = [];
nan_params = [];

PVE = [];
EE = [];
EEOff = [];

l4_counter = [];
l23_counter = [];
l56_counter = [];
nan_counter = [];

idx_tracker = [];

all_params = [];
colors = [];
colors2 = [];
for k = 1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    data  = load(fpath);                          % loads all vars into a struct
    
    %Find min loss
    losses = squeeze(data.losses(:,2,:));
    min_losses = min(min(losses));
    idx_val = find(losses==min_losses);
    
    %Dont allow for ties
    sizes = size(losses);

    index2 = ceil(idx_val(1)/sizes(1));
    index1 = idx_val(1) - ((index2-1)*sizes(1));

    all_params = [all_params;data.param_tracker(index1,:,index2)];
    %Color by layer
    colors = [colors;cmap(find(layers == all_data(k).layer),:)];
    
    c_val = 1;

    if label_struct.our_struct(k).n90.is_Both == 1
        c_val = 3;
    elseif label_struct.our_struct(k).n90.is_Neither == 1
         c_val = 4;
    elseif label_struct.our_struct(k).n90.is_Offset == 1
         c_val = 2;
    end

    colors2 = [colors2;cmap(c_val,:)];

    if idx4(k)
        l4_params = [l4_params, data.param_tracker(index1,4,index2)];
    elseif idx56(k)
        l56_params = [l56_params, data.param_tracker(index1,4,index2)];
    elseif idx23(k)
        l23_params = [l23_params, data.param_tracker(index1,4,index2)];
    elseif idxnan(k)
        nan_params = [nan_params, data.param_tracker(index1,4,index2)];
    end

    PVE = [PVE,data.param_tracker(index1,4,index2)];
    EE = [EE,data.param_tracker(index1,1,index2)];
    EEOff = [EEOff,data.param_tracker(index1,2,index2)];

    if data.param_tracker(index1,4,index2) > data.param_tracker(index1,1,index2)
        if idx4(k)
            l4_counter = [l4_counter, 1];
        elseif idx56(k)
            l56_counter = [l56_counter, 1];
        elseif idx23(k)
            l23_counter = [l23_counter , 1];
        elseif idxnan(k)
            nan_counter = [nan_counter, 1];
        end
    end

    if data.param_tracker(index1,4,index2) < data.param_tracker(index1,1,index2)
        if idx4(k)
            l4_counter = [l4_counter, 0];
        elseif idx56(k)
            l56_counter = [l56_counter, 0];
        elseif idx23(k)
            l23_counter = [l23_counter , 0];
        elseif idxnan(k)
            nan_counter = [nan_counter, 0];
        end
    end
    idx_tracker = [idx_tracker, k];


end

%%



X = all_params;
Xmax = max(X);
zero1 = X./Xmax;   
Xc = zero1 - mean(zero1,1);
C = (Xc.'*Xc)/(size(X,1)-1);   % D x D covariance
[vecs, vals] = eig(C, 'vector');
[latent, order] = sort(vals, 'descend');
coeff = vecs(:,order);
score = Xc * coeff;
explained = 100 * latent / sum(latent);

figure;
%s = scatter(score(:,10), score(:,11), 50, colors2, 'filled');

axis_pca = score(:,11);
%%
close all
figure("Position",[200,300,1000,500]);
%subplot(4,4,[1,2;5,6;9,10;13,14])
t = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
ax = nexttile(t,[4 2]);
%s = scatter(axis_pca,zeros(length(score(:,11)),1), 50, colors2, 'filled');
s = scatter(all_params(:,1),all_params(:,2), 50, colors2, 'filled');
xlabel(ax,'On\rightarrow E strength','FontWeight','bold','Interpreter','tex')
ylabel(ax,'Off\rightarrow E strength','FontWeight','bold','Interpreter','tex')


% for i = 1:length(all_params(:,1))
%     text(all_params(i,1), all_params(i,2), sprintf('%d', i), ...
%     'VerticalAlignment', 'bottom', ...
%     'HorizontalAlignment', 'right', ...
%     'FontSize', 8, ...
%     'Color', 'k');
% end

grid on; set(gcf,'Color','w'); hold on;

% Hide the main scatter from the legend
set(s, 'HandleVisibility','off');

% Make legend proxies that match your cmap/layers
layers = ["L2/3","L4","L5/6","NaN"];
cmap  = [ 57 106 177;
          62 150  81;
         204  37  41;
         160 160 160 ]/255;

h = gobjects(numel(layers),1);

figure;
heatmap(coeff);
colormap('parula');



%Plot individual cells
point1 = 148;
point2 = 74;

%subplot(4,4,[3,4])
nexttile(t,[1 2]);
bin_width = 200; %In 0.1 ms

%Use the same PSTH as used in the training

fpath = fullfile(files(point1).folder, files(point1).name);
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
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

%plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
plot(sim_PSTH)
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Sim Raster: Cell ' + string(point1))

%subplot(4,4,[7,8])
nexttile(t,[1 2]);
data = load('picture_fit' + string(point1) + 'contra.mat').picture;

num_bins = floor(size(data,2)/bin_width);
remainder = mod(size(data,2),bin_width);

sim_trim = data(:,remainder+1:end); 

index_course = 1:size(sim_trim,2); 

sim_scaled = sim_trim.*index_course;
bin_edges = 1:bin_width:size(sim_scaled,2)+1;
data_PSTH = histcounts(sim_scaled,bin_edges);
[trial_sim,indicy_sim] = find(sim_scaled);
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

[r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
[rmax, idx] = max(r);

%plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
plot(data_PSTH)
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Data Raster: Cell ' + string(point1) + ' Max Xcorr = ' + string(rmax))

%subplot(4,4,[11,12])
nexttile(t,[1 2]);
fpath = fullfile(files(point2).folder, files(point2).name);
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
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

%plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
plot(sim_PSTH)
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Sim Raster: Cell ' + string(point2))

%subplot(4,4,[15,16])
nexttile(t,[1 2]);
data = load('picture_fit' + string(point2) + 'contra.mat').picture;
num_bins = floor(size(data,2)/bin_width);
remainder = mod(size(data,2),bin_width);

sim_trim = data(:,remainder+1:end); 

index_course = 1:size(sim_trim,2); 


sim_scaled = sim_trim.*index_course;
bin_edges = 1:bin_width:size(sim_scaled,2)+1;
data_PSTH = histcounts(sim_scaled,bin_edges);
[trial_sim,indicy_sim] = find(sim_scaled);
entries_data = [indicy_sim,indicy_sim]';
tick_data = [trial_sim,trial_sim-1]'; %Trial and Trial-1

[r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
[rmax, idx] = max(r);

%plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
plot(data_PSTH)
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Data Raster: Cell ' + string(point2) + ' Max Xcorr = ' + string(rmax))


load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

figure;
plot(target1)



figure("Position",[200,300,1000,300]);
%subplot(4,4,[1,2;5,6;9,10;13,14])
t = tiledlayout(6,4,'TileSpacing','compact','Padding','compact');
ax = nexttile(t,[2 2]);
plot(entries_data,tick_data,'k',LineWidth=2); hold on;
xticklabels('')
yticklabels('')
title('Data Raster','FontSize',12)
ax = nexttile(t,[6 2]);
plot(data_PSTH,LineWidth=2); hold on
plot(sim_PSTH,LineWidth=2); hold on
xticklabels('')
yticklabels('')
legend({'Data','Sim'},'FontSize',12)
title('PSTH Comparison','FontSize',11)
ax = nexttile(t,[2 2]);
plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
xticklabels('')
yticklabels('')
title('Simulation Raster','FontSize',13)
ax = nexttile(t,[2 2]);
plot(target1)
title('Stimulus')
xticklabels('')
yticklabels('')

% (1) Avoid transparency – it forces rasterization
%set(findobj(gcf,'Type','Scatter'), 'MarkerFaceAlpha',1, 'MarkerEdgeAlpha',1);

% (2) Force vector renderer
%set(gcf,'Renderer','painters');

% (3) Write SVG as vectors
%print(gcf, '-dsvg', '-painters', 'figure2.svg');   % add -r600 if you have any images

