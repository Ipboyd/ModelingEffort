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

rmaxs = [];
R2s = [];

for k = 1:220
    fpath = fullfile(files(k).folder, files(k).name);
    sim_load  = load(fpath);
    sim = sim_load.best_output;

    bin_width = 200; %In 0.1 ms

    data = load('picture_fit' + string(k) + 'contra.mat').picture;

    num_bins = floor(size(data,2)/bin_width);
    remainder = mod(size(data,2),bin_width);
    
    data_trim = data(:,remainder+1:end); 
    sim_trim = sim(:,remainder+1:end); 

    index_course = 1:size(data_trim,2); 
    
    data_scaled = data_trim.*index_course;
    sim_scaled = sim_trim.*index_course;
    bin_edges = 1:bin_width:size(sim_scaled,2)+1;
    data_PSTH = histcounts(data_scaled,bin_edges);
    sim_PSTH = histcounts(sim_scaled,bin_edges);
    % [trial_data,indicy_data] = find(data_scaled);
    % [trial_sim,indicy_sim] = find(sim_scaled);
    % 
    % entries_sim = [indicy_sim,indicy_sim]';
    % tick_sim = [trial_sim,trial_sim-1]'; 
    % 
    % entries_data = [indicy_data,indicy_data]';
    % tick_data = [trial_data,trial_data-1]'; 

    [r, lags] = xcorr(sim_PSTH, data_PSTH, 'coeff');
    [rmax, idx] = max(r);

    rmaxs = [rmaxs, rmax];

    %SS_res = sum((sim_PSTH - data_PSTH).^2);
    %SS_tot = sum((data_PSTH - mean(data_PSTH)).^2);
    %R2 = 1 - SS_res / SS_tot;

    %R2s = [R2s, R2];

    mdl = fitlm(sim_PSTH, data_PSTH);
    R2 = mdl.Rsquared.Ordinary;
    
    R2s = [R2s, R2];

    %figure;
    %scatter(sim_PSTH,data_PSTH)

end
disp('mean Xcorr')
disp(mean(rmaxs))
disp('mean R2')
disp(mean(R2s))