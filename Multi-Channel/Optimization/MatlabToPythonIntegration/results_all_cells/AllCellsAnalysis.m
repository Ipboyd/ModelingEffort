addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting\PicturesToFit')

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');

folder = 'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells_both';
files  = dir(fullfile(folder,'*.mat'));
[~,idx] = sort([files.datenum]);  % optional: sort by time

%The following fixes this. Comment out if you do another new run.
idx = idx - 1; %Shift everything one over
idx(1) = 122; %Make the 122nd the first.
idx(123:220) = 123:220;

files = files(idx);



ifplot = 1;

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
    
    
    %Find all the fits in the top x%
    %thr = prctile(reshape(losses,[1,8000]), 0.1);
    %idx_val = find(losses<=thr);

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

    %Color by FR
    %colors = [colors;length(vertcat(all_data(k).ctrl_tar1_timestamps{:,1}))];
    %Color by loss?
    %colors = [colors;data.losses(index1,2,index2)];
    if idx4(k)
        l4_params = [l4_params, data.param_tracker(index1,4,index2) - ((data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2)))];
    elseif idx56(k)
        l56_params = [l56_params, data.param_tracker(index1,4,index2)- ((data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2)))];
    elseif idx23(k)
        l23_params = [l23_params, data.param_tracker(index1,4,index2)- ((data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2)))];
    elseif idxnan(k)
        nan_params = [nan_params, data.param_tracker(index1,4,index2)- ((data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2)))];
    end

    PVE = [PVE,data.param_tracker(index1,4,index2)];
    EE = [EE,data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2)];
    EEOff = [EEOff,data.param_tracker(index1,2,index2)];

    if data.param_tracker(index1,4,index2) > (data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2))
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

    if data.param_tracker(index1,4,index2) < (data.param_tracker(index1,1,index2) + data.param_tracker(index1,2,index2))
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

    % %Allow for ties
    % for t = 1:length(idx_val)
    % 
    %     sizes = size(losses);
    % 
    %     index2 = ceil(idx_val(t)/sizes(1));
    %     index1 = idx_val(t) - ((index2-1)*sizes(1));
    % 
    %     all_params = [all_params;data.param_tracker(index1,:,index2)];
    %     %Color by layer
    %     colors = [colors;cmap(find(layers == all_data(k).layer),:)];
    %     %Color by FR
    %     %colors = [colors;length(vertcat(all_data(k).ctrl_tar1_timestamps{:,1}))];
    %     %Color by loss?
    %     %colors = [colors;data.losses(index1,2,index2)];
    %     if idx4(k)
    %         l4_params = [l4_params, data.param_tracker(index1,10,index2)];
    %     elseif idx56(k)
    %         l56_params = [l56_params, data.param_tracker(index1,10,index2)];
    %     elseif idx23(k)
    %         l23_params = [l23_params, data.param_tracker(index1,10,index2)];
    %     elseif idxnan(k)
    %         nan_params = [nan_params, data.param_tracker(index1,10,index2)];
    %     end
    % 
    %     PVE = [PVE,data.param_tracker(index1,3,index2)];
    %     EE = [EE,data.param_tracker(index1,1,index2)];
    % 
    %     if data.param_tracker(index1,3,index2) > data.param_tracker(index1,1,index2)
    %         if idx4(k)
    %             l4_counter = [l4_counter, 1];
    %         elseif idx56(k)
    %             l56_counter = [l56_counter, 1];
    %         elseif idx23(k)
    %             l23_counter = [l23_counter , 1];
    %         elseif idxnan(k)
    %             nan_counter = [nan_counter, 1];
    %         end
    %     end
    % 
    %     if data.param_tracker(index1,3,index2) < data.param_tracker(index1,1,index2)
    %         if idx4(k)
    %             l4_counter = [l4_counter, 0];
    %         elseif idx56(k)
    %             l56_counter = [l56_counter, 0];
    %         elseif idx23(k)
    %             l23_counter = [l23_counter , 0];
    %         elseif idxnan(k)
    %             nan_counter = [nan_counter, 0];
    %         end
    %     end
    %     idx_tracker = [idx_tracker, k];
    % end

end

%% Load in Layer Data
%cd(userpath);
%cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')


%%

%Get E->PV gsyn
%all_params(:,3)
figure;
scatter(EE,PVE,36,colors,'filled', 'MarkerEdgeColor',[0 0 0], ...
        'MarkerFaceAlpha',0.85, 'MarkerEdgeAlpha',0.5); hold on
plot(linspace(0,0.1),linspace(0,0.1),'k--')
xlabel('E->E strength')
ylabel('PV->E strength')

for i = 1:length(EE)
    text(EE(i), PVE(i), sprintf('%d', i), ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right', ...
        'FontSize', 8, ...
        'Color', 'k');
end


disp('Fraction for each cell type where PV->E > E->E')
disp('Layer 4')
disp(sum(l4_counter)/length(l4_counter))
disp('Layer 23')
disp(sum(l23_counter)/length(l23_counter))
disp('Layer 56')
disp(sum(l56_counter)/length(l56_counter))
disp('Layer nan')
disp(sum(nan_counter)/length(nan_counter))
%%

figure("Position",[200,300,1000,500]);
%subplot(4,4,[1,2;5,6;9,10;13,14])
t = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
ax = nexttile(t,[4 2]);
scatter(EE,PVE,75,colors,'filled', 'MarkerEdgeColor',[0 0 0], ...
        'MarkerFaceAlpha',0.6, 'MarkerEdgeAlpha',0.5); hold on
plot(linspace(-0.002,0.1,10),linspace(-0.002,0.1,10),'k--','LineWidth',2)
xlabel(ax,'E\rightarrow E strength','FontWeight','bold','Interpreter','tex')
ylabel(ax,'PV\rightarrow E strength','FontWeight','bold','Interpreter','tex')
ax.FontSize = 16;

legendNames = {'L2/3','L4','L5/6','NaN'};   % <-- your labels in the same order as cmap rows
proxy = gobjects(numel(legendNames),1);
for i = 1:numel(legendNames)
    proxy(i) = scatter(ax, NaN, NaN, 75, cmap(i,:), 'filled', ...
        'MarkerEdgeColor',[0 0 0], 'DisplayName', legendNames{i});
end
lgd = legend(ax, proxy, legendNames, 'Box','off','Location','northeast','FontSize',16);

xlim([-0.002,0.14])
ylim([-0.002,0.1])

ax = nexttile(t,[4 2]);
vals = [sum(l23_counter)/numel(l23_counter),
        sum(l4_counter)/numel(l4_counter),
        sum(l56_counter)/numel(l56_counter)];

X = categorical({'L2/3','L4','L5/6'});
X = reordercats(X, {'L2/3','L4','L5/6'});  % keep order you want

b = bar(X, vals, 'FaceColor','flat');
b.CData = cmap(1:3,:);
yticks(0.1:0.2:1.5)
ax.FontSize = 16;



ylim([0, 1])
%yticks('')
%%
figure("Position",[200,300,1000,500]);
%subplot(4,4,[1,2;5,6;9,10;13,14])
t = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
ax = nexttile(t,[4 2]);
scatter(EE,PVE,75,colors,'filled', 'MarkerEdgeColor',[0 0 0], ...
        'MarkerFaceAlpha',0.6, 'MarkerEdgeAlpha',0.5); hold on
plot(linspace(-0.01,0.14),linspace(-0.01,0.14),'k--','LineWidth',2)
xlabel(ax,'E\rightarrow E strength','FontWeight','bold','Interpreter','tex')
ylabel(ax,'PV\rightarrow E strength','FontWeight','bold','Interpreter','tex')



% for i = 1:length(EE)
%     text(EE(i), PVE(i), sprintf('%d', i), ...
%         'VerticalAlignment', 'bottom', ...
%         'HorizontalAlignment', 'right', ...
%         'FontSize', 8, ...
%         'Color', 'k');
% end



%disp(sum(l4_counter)/length(l4_counter))
%disp(sum(l23_counter)/length(l23_counter))
%disp(sum(l56_counter)/length(l56_counter))
%disp(sum(nan_counter)/length(nan_counter))


point1 = idx_tracker(47);
point2 = idx_tracker(1);

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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

[r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
[rmax, idx] = max(r);

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Data Raster: Cell ' + string(point2) + ' Max Xcorr = ' + string(rmax))

%all_data.layer
if ifplot == 2
    num_bins = 100;
    
    figure;
    subplot(2,2,1);
    histogram(l4_params,num_bins)
    subplot(2,2,2);
    histogram(l23_params,num_bins)
    subplot(2,2,3);
    histogram(l56_params,num_bins)
    subplot(2,2,4);
    histogram(nan_params,num_bins)
    
    mean(l4_params)
    mean(l23_params)
    mean(l56_params)
    mean(nan_params)
    
    % Colors (same order you used before: L2/3, L4, L5/6, NaN)
    cmap = [ 57 106 177;
             62 150  81;
            204  37  41;
            160 160 160 ]/255;
    
    % Common bin edges so shapes are comparable
    all_vals = [l23_params(:); l4_params(:); l56_params(:); nan_params(:)];
    edges = linspace(min(all_vals), max(all_vals), num_bins+1);
    
    figure; hold on; grid on; set(gcf,'Color','w');
    
    [f,x] = ksdensity(l23_params,'NumPoints',num_bins); area(x,f,'FaceColor',cmap(1,:), 'FaceAlpha',0.25, 'EdgeColor','none');
    [f,x] = ksdensity(l4_params,'NumPoints',num_bins);  area(x,f,'FaceColor',cmap(2,:), 'FaceAlpha',0.25, 'EdgeColor','none');
    [f,x] = ksdensity(l56_params,'NumPoints',num_bins); area(x,f,'FaceColor',cmap(3,:), 'FaceAlpha',0.25, 'EdgeColor','none');
    [f,x] = ksdensity(nan_params,'NumPoints',num_bins); area(x,f,'FaceColor',cmap(4,:), 'FaceAlpha',0.25, 'EdgeColor','none');
    
    xlabel('Parameter value'); ylabel('Probability density');
    title('KDE overlays by layer'); legend({'L2/3','L4','L5/6','NaN'}, 'Location','best', 'Box','off');
end
%%
if ifplot == 1
    X = all_params;
    
    Xmax = max(X);
    zero1 = X./Xmax;
    
    

    %Xc = X - mean(X,1);
    
    Xc = zero1 - mean(zero1,1);
    

    C = (Xc.'*Xc)/(size(X,1)-1);   % D x D covariance
    [vecs, vals] = eig(C, 'vector');
    [latent, order] = sort(vals, 'descend');
    coeff = vecs(:,order);
    score = Xc * coeff;
    explained = 100 * latent / sum(latent);
    
    figure;
    %s = scatter(score(:,10), score(:,11), 50, colors2, 'filled');

    s = scatter(score(:,11),zeros(length(score(:,11)),1), 50, colors2, 'filled');
    grid on; set(gcf,'Color','w'); hold on;
    %colormap(turbo); colorbar; grid on; set(gcf,'Color','w');
    %caxis([min(colors) max(colors)]); 
    
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
    colormap('parula')

    % for i = 1:numel(layers)
    %     % dummy points just for legend, colored like the real data
    %     h(i) = plot(nan, nan, 'o', ...
    %         'MarkerFaceColor', cmap(i,:), ...
    %         'MarkerEdgeColor', 'none', ...
    %         'MarkerSize', 10, ...
    %         'DisplayName', layers(i));
    % end



    
    %legend(h, 'Location','northeast' , 'Box','on','FontSize',20);  % or 'NumColumns',2
    %xlabel('PC1'); ylabel('PC2'); title('Units colored by layer');
end

%%

if ifplot == 2
    %%
    %Stat test
    G = {l23_params(:), l4_params(:), l56_params(:), nan_params(:)};
    names = ["L2/3","L4","L5/6","NaN"];
    
    % Omnibus nonparametric test
    x = cell2mat(G');                      % stack
    g = arrayfun(@(i) repmat(names(i), numel(G{i}), 1), 1:numel(G), 'uni', false);
    g = vertcat(g{:});
    p_kw = kruskalwallis(x, g, 'on');     % 'off' = no plot
    fprintf('Kruskal–Wallis p = %.3g\n', p_kw);
    
    % Post-hoc pairwise with built-in correction
    if p_kw < 0.05
        [~,~,stats] = kruskalwallis(x, g, 'off');
        C = multcompare(stats, 'ctype','dunn-sidak','display','off');  % pairs & adjusted p
        % C columns: [i j lower meanDiff upper pAdj]
        for r = 1:size(C,1)
            i = C(r,1); j = C(r,2);
            fprintf('%s vs %s: adj p = %.3g\n', names(i), names(j), C(r,6));
        end
    end
end




%Scatter for onsets and offsets

figure("Position",[200,300,1000,500]);
%subplot(4,4,[1,2;5,6;9,10;13,14])
t = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
ax = nexttile(t,[4 2]);
scatter(EE,EEOff,75,colors2,'filled', 'MarkerEdgeColor',[0 0 0], ...
        'MarkerFaceAlpha',0.6, 'MarkerEdgeAlpha',0.5); hold on
plot(linspace(0,0.1),linspace(0,0.1),'k--','LineWidth',2)
xlabel(ax,'ON\rightarrow E strength','FontWeight','bold','Interpreter','tex')
ylabel(ax,'OFF\rightarrow E strength','FontWeight','bold','Interpreter','tex')

% for i = 1:length(EE)
%     text(EE(i), PVE(i), sprintf('%d', i), ...
%         'VerticalAlignment', 'bottom', ...
%         'HorizontalAlignment', 'right', ...
%         'FontSize', 8, ...
%         'Color', 'k');
% end



%disp(sum(l4_counter)/length(l4_counter))
%disp(sum(l23_counter)/length(l23_counter))
%disp(sum(l56_counter)/length(l56_counter))
%disp(sum(nan_counter)/length(nan_counter))


point1 = idx_tracker(47);
point2 = idx_tracker(1);

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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
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
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

[r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
[rmax, idx] = max(r);

plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Data Raster: Cell ' + string(point2) + ' Max Xcorr = ' + string(rmax))


%%
figure;
subplot(3,1,1)
histogram(l4_params,20)
xlim([-0.07,0.09])
subplot(3,1,2)
histogram(l23_params,20)
xlim([-0.07,0.09])
subplot(3,1,3)
histogram(l56_params,20)
xlim([-0.07,0.09])