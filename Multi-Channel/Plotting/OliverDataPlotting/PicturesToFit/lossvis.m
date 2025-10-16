addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_arch_test\')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_tau_ad\')
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting\PicturesToFit')
m = matfile('run_2025-10-15_20-05-36.mat');
%m = matfile('run_2025-10-09_05-49-04.mat');

loss = mean(losses(:,2,:),3);
min_loss = min(squeeze(losses(:,2,:))');

figure;
subplot(3,1,1)
plot(loss(50:end)); hold on
plot(movmean(loss(50:end),10))
xlim([-50,250])
subplot(3,1,2)
plot(min_loss);hold on
plot(movmean(min_loss,10))
subplot(3,1,3)
plot(movmean(min_loss,10))



bin_width = 200
sim = m.best_output;

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


data = load('picture_fit' + string(7) + 'contra.mat').picture;

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


figure;
subplot(2,1,1)
plot(entries_sim,tick_sim,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Sim Raster: Cell ' + string(7))

subplot(2,1,2)
plot(entries_data,tick_data,'k',LineWidth=2); hold on;
yticklabels('')
xticklabels('')
xlabel('Time')
ylabel('Trial')
title('Sim Raster: Cell ' + string(7))