function [performance, tau] = mouse_network(study_dir,time_end,plot_rasters)
%% pseudocode:

% load spikes from mouse model - generated from...?
%   mouse model mat file contents:
%       songloc
%       maskerloc
%       sigma (tuning curve width)
%       paramH ?
%       paramG ?
%       rand_seed
%       mean_rate
%       t_spiketimes in ms, 10 trials per song, for 2 songs
%       spkrate
%       disc
% spikes processed with lateral inhibition network
% compare output spikes to calculate discriminality measure
%% solver params
% time_end = 1860; % ms, must be smaller than time-data

solverType = 'euler';
dt = 1; %ms % the IC input is currently at dt=1


% visualize IC spikes (Figure 1 which is the IR level (as seen from the
% inputguassian file)

%{
figure
for i = 1:4 %for each spatially directed neuron
   subplot(1,4,i)
   plotSpikeRasterFs(logical(squeeze(spks(:,i,:))), 'PlotType','vertline');
   xlim([0 2000])
   line([0,2000],[10.5,10.5],'color',[0.3 0.3 0.3])
end
%}

% save converted spike file - only saving spikes the last file for now
% % save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');
%% neuron populations

nCells = 4;
noise = 0.01; % low noise

s = struct();

% neuron populations
s.populations(1).name = 'IC';
s.populations(1).equations = 'chouLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'Itonic',0,'noise',0}; % 10-20 Hz spiking at rest

s.populations(end+1).name = 'I';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'Itonic',0, 'noise',noise}; % 10-20 Hz spiking at rest
%tonic = bias - cells spontaneous firing

s.populations(end+1).name='R';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'Itonic',0,'noise',noise};

s.populations(end+1).name='C';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = 1;
s.populations(end).parameters = {'noise',noise};

%% connection mechanisms
synDoubleExp={
  'gSYN = 1; ESYN = 0; tauD = 4; tauR = 1; delay = 0'
  'f(x) = (exp(-x/tauD) - exp(-x/tauR)).*(x>0)'
  'netcon=ones(N_pre,N_post)'
  'synDoubleExp(X,t) = gSYN .* ( f(t - tspike_pre - delay) * netcon ).*(X - ESYN)'
  '@isyn += synDoubleExp(V_post,t)'
  };

s.mechanisms(1).name='synDoubleExp';
s.mechanisms(1).equations=synDoubleExp;

%% connections
% build I->R netcon matrix
% netcons are [N_pre,N_post]
% irNetcon = diag(ones(1,nCells));
irNetcon = zeros(nCells);
irNetcon(2,1) = 1;
irNetcon(3,1) = 1;
irNetcon(4,1) = 1;
irNetcon(2,4) = 1;

% irNetcon(3,:) = 0;
% irNetcon(3,3) = 0;

s.connections(1).direction='IC->IC';
s.connections(1).mechanism_list='IC';
s.connections(1).parameters={'g_postIC',0.03,'trial',5}; % 100 hz spiking

s.connections(end+1).direction='IC->I';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.12, 'tauR',0.4, 'tauD',2, 'netcon',diag(ones(1,nCells))}; 

s.connections(end+1).direction='IC->R';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.12, 'tauR',0.4, 'tauD',2, 'netcon',diag(ones(1,nCells))}; 

s.connections(end+1).direction='I->R';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.25, 'tauR',0.4, 'tauD',10, 'netcon',irNetcon, 'ESYN',-80}; 

s.connections(end+1).direction='R->C';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.13, 'tauR',0.4, 'tauD',2, 'netcon','ones(N_pre,N_post)'}; 

%% vary params
vary = {
  '(IC->IC)', 'trial', 1:20;
%     'I->R','gSYN',.1:.05:.75;
};
%% simulate
tic;
data = dsSimulate(s,'time_limits',[dt time_end], 'solver',solverType, 'dt',dt,...
  'downsample_factor',1, 'save_data_flag',1, 'save_results_flag',1,...
  'study_dir',study_dir, 'vary',vary, 'debug_flag',1, 'verbose_flag',0);
toc
%% insert spikes
V_spike = 50;
for iData = 1:length(data)
  for pop = {s.populations.name}
    pop = pop{1};
    data(iData).([pop '_V'])(data(iData).([pop '_V_spikes']) == 1) = V_spike; % insert spike
  end
end
%% visualize spikes
if plot_rasters
ICspks = zeros(20,4,time_end);
Ispks = zeros(20,4,time_end);
Rspks = zeros(20,4,time_end);
for i = 1:20
    for j = 1:4
        ICspks(i,j,:) = data(i).IC_V_spikes(:,j);
        Ispks(i,j,:) = data(i).I_V_spikes(:,j);
        Rspks(i,j,:) = data(i).R_V_spikes(:,j);
    end
end
Cspks = [data.C_V_spikes];

% plot
figure
for i = 1:4 %for each spatially directed neuron
  subplot(4,4,i+12)
  plotSpikeRasterFs(logical(squeeze(ICspks(:,i,:))), 'PlotType','vertline');
  xlim([0 2000])
  line([0,2000],[10.5,10.5],'color',[0.3 0.3 0.3])
  if i==1, ylabel('IC'); 
end
  
  subplot(4,4,i+8)
  plotSpikeRasterFs(logical(squeeze(Ispks(:,i,:))), 'PlotType','vertline');
  xlim([0 2000])
  line([0,2000],[10.5,10.5],'color',[0.3 0.3 0.3])
  if i==1, ylabel('I'); end
    
  subplot(4,4,i+4)
  plotSpikeRasterFs(logical(squeeze(Rspks(:,i,:))), 'PlotType','vertline');
  xlim([0 2000])
  line([0,2000],[10.5,10.5],'color',[0.3 0.3 0.3])
  if i==1, ylabel('R'); end
end

subplot(4,4,2)
plotSpikeRasterFs(logical(Cspks'), 'PlotType','vertline');
xlim([0 2000])
line([0,2000],[10.5,10.5],'color',[0.3 0.3 0.3])
ylabel('C spikes')
end
%% spks to spiketimes in a cell array of 10x8
for i = 1:20
    if i <=10
        spkTimes{i,1} = find(data(i).C_V_spikes);
    else
        spkTimes{i-10,2} = find(data(i).C_V_spikes);
    end
end

%% performance for the current target-masker config
%addpath('C:\Users\Kenny\Dropbox\Sen Lab\MouseSpatialGrid\spatialgrids')
tau= linspace(4,1000,1000); %same units as spike-timing
distMat = calcvr(spkTimes, tau);
[performance, E] = calcpc(distMat, 10, 2, 1,[], 'new');
