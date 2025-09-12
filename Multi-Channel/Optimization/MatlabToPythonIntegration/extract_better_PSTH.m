close all;

filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/picture_fit.mat";
data = load(filename).picture;


%PSTH + VR 100ms
m = matfile('run_2025-09-11_20-28-42.mat');


outputs = m.output;
losses = m.losses;
params = m.param_tracker;

%PSTH only
%m = matfile('run_2025-09-05_22-19-39.mat');

[min_val, min_idx] = min(losses(300,2,:));

output_trial = min_idx;
bin_width = 200; %In 0.1 ms

%Use the same PSTH as used in the training
num_bins = floor(size(data,2)/bin_width);
remainder = mod(size(data,2),bin_width);
data_trimmed = data(:,remainder+1:end); %remaineder + 1 due to 1 indexing in matlab
index_course = 1:size(data_trimmed,2); %Multipy by the index because histcounts looks at the # of times a number (index) occurs within a window
data_scaled = data_trimmed.*index_course;
bin_edges = 1:bin_width:size(data_scaled,2)+1;
data_PSTH = histcounts(data_scaled,bin_edges);

%Compare to a given sim PSTH
sim = outputs(:,:,output_trial);
sim_trim = sim(:,remainder+1:end,:); 
sim_scaled = sim_trim.*index_course;
sim_PSTH = histcounts(sim_scaled,bin_edges);

%Gather the nonzero entires to plot the raster
[trial,indicy] = find(data_scaled);
entries = [indicy,indicy]';
tick = [trial,trial-1]'; %Trial and Trial-1

[trial_sim,indicy_sim] = find(sim_scaled);
entries_sim = [indicy_sim,indicy_sim]';
tick_sim = [trial_sim,trial_sim-1]'; %Trial and Trial-1

figure(1);
set(gcf, 'Position', [300, 500, 600, 400]);
subplot(2,1,1)
plot(entries,tick,'k',LineWidth=2);
yticklabels('')
xticklabels('')
ylabel('Trial')
title('Data Raster')

subplot(2,1,2)
plot(entries_sim,tick_sim,'k',LineWidth=2);
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Sim Raster')

loss_val = sum((data_PSTH-sim_PSTH).^2);

figure(2)
set(gcf, 'Position', [1000, 500, 600, 400]);
plot(data_PSTH,'b',LineWidth=2);hold on
plot(sim_PSTH,'r',LineWidth=2); hold off
sgtitle('PSTH Comparsion -- Loss: ' + string(loss_val))
yticklabels('')
xticklabels('')
legend({'data','sim'},FontSize=15)




