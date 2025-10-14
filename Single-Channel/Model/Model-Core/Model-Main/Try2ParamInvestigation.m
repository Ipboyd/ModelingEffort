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
idx(123:220) = 123:220;

%%

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

all_params = [];
for k = 1:numel(files)
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
end

%Extract out the params for each layer

%% HeatMaps for layers

l4_params = all_params(idx4,:);
l56_params = all_params(idx56,:);
l23_params = all_params(idx23,:);



close all
figure;
heatmap(l4_params(:,1:5))
title('Layer4')
disp(sum(l4_params(:,1:5))/length(l4_params(:,1:5)))
figure;
heatmap(l23_params(:,1:5))
title('Layer23')
disp(sum(l23_params(:,1:5))/length(l23_params(:,1:5)))
figure;
heatmap(l56_params(:,1:5))
title('Layer56')
disp(sum(l56_params(:,1:5))/length(l56_params(:,1:5)))

%% Swarm Plot for layers
close all
for j = 1:11
    %j = 11; % pick a parameter index
    data = [l4_params(:,j); l23_params(:,j); l56_params(:,j)];
    G = [ones(size(l4_params,1),1); 2*ones(size(l23_params,1),1); 3*ones(size(l56_params,1),1)];
    grp  = categorical(G);
    figure;
    violinplot(G, data); hold on;

    swarmchart(G, data, 36, 'filled'); box off; ylabel((j));
end


%% Distribution of relative weight violin/swarm

data = [l4_params(:,4)-(l4_params(:,1) + l4_params(:,2)); l23_params(:,4)-(l23_params(:,1) + l23_params(:,2)); l56_params(:,4)-(l56_params(:,1) + l56_params(:,2))];
G = [ones(size(l4_params,1),1); 2*ones(size(l23_params,1),1); 3*ones(size(l56_params,1),1)];
grp  = categorical(G);
figure;
violinplot(G, data); hold on;

swarmchart(G, data, 36, 'filled'); box off; ylabel((j));

%% Stat test
% Prepare your data
data = [l4_params(:,4)-(l4_params(:,1) + l4_params(:,2)); 
        l23_params(:,4)-(l23_params(:,1) + l23_params(:,2)); 
        l56_params(:,4)-(l56_params(:,1) + l56_params(:,2))];

% Grouping variable
G = [ones(size(l4_params,1),1); 
     2*ones(size(l23_params,1),1); 
     3*ones(size(l56_params,1),1)];

% Perform the Kruskal-Wallis test
[p, tbl, statsk] = kruskalwallis(data, G);
disp(['P-value from Kruskal-Wallis test: ', num2str(p)]);

% Prepare your data
data = [l4_params(:,4)-(l4_params(:,1) + l4_params(:,2)), 
        l23_params(:,4)-(l23_params(:,1) + l23_params(:,2)), 
        l56_params(:,4)-(l56_params(:,1) + l56_params(:,2))];

% Perform one-way ANOVA
[p, tbl, stats] = anova1(data, G);
disp(['P-value from ANOVA: ', num2str(p)]);

C_anova = multcompare(stats, 'CType','tukey-kramer', 'Display','off'); 
%%
C_kw = multcompare(statsk, 'CType','dunn-sidak', 'Display','off');