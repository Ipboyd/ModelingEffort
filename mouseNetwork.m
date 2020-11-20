function [simdata,s] = mouseNetwork(study_dir,varies,netcons,restricts,options)
% [perf, fr, annotstr, distMat, VR] = mouse_network(study_dir,time_end,...
%    varies,plot_rasters,plot_distances,data_spks,data_tau,restrict_vary_flag)

% [performance, tauMax] = mouse_network(study_dir,time_end,varies,plot_rasters,data_spks)
% study_dir: location of IC spike files + directory for log and data files
% time_end: length of simulation in ms
% varies: vary parameter, a structure. e.g.
%   varies(1).conxn = '(IC->IC)';
%   varies(1).param = 'trial';
%   varies(1).range = 1:20;
%   *first set of parameters should always be "trial" for IC->IC cxn
% plot_rasters: 1 or 0
% data_spks: spikes from data we want to match 
%
% @Kenny F Chou, Boston Univ. 2019-06-20
% 2019-08-04 - added sharpening neurons
% 2019-08-14 - plotting now handles multiple varied parameters
% 2019-09-11 - removed redundant code. Return R performance in addition to
%              C performance
% 2020-02-05 - added netCons as parameter
% to do - fix plot_rasters option

% @Jio Nocon, Boston Univ., 2020-6-18
% 2020-6-18 - added subfunction to plot model rasters vs. data rasters and
%             calculate VR distance between the two

% @Jio Nocon, BU 2020-10-14
% 2020-10-14 - split inputs to R and S into EIC and IIC, respectively

% 2020-11-17 - KC - general cleanup

%% Input check
if ~strcmp(varies(1).param,'trial')
    error('first set of varied params should be ''trial''')
end

%% solver params
solverType = 'euler';
dt = 1; %ms % the IC input is currently dt=1
time_end = options.time_end;

%% neuron populations
% tonic = bias = cells spontaneous firing

nCells = 4;
s = struct();

s.populations(1).name = 'Exc';
s.populations(1).equations = 'chouLIF';
s.populations(1).size = nCells;

s.populations(end+1).name = 'Inh';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;

s.populations(end+1).name = 'X';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;

s.populations(end+1).name='R';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;

s.populations(end+1).name='C';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = 1;
s.populations(end).parameters = {'Ad_tau',0};

%% connections
XRnetcon = netcons.xrNetcon;

s.connections(1).direction='Exc->Exc';
s.connections(1).mechanism_list={'IC'};
s.connections(1).parameters={'g_postIC',0.25,'label','E','ICdir',options.ICdir,'locNum',options.locNum}; % 100 hz spiking

s.connections(end+1).direction='Inh->Inh';
s.connections(end).mechanism_list={'IC'};
s.connections(end).parameters={'g_postIC',0.25,'label','I','ICdir',options.ICdir,'locNum',options.locNum}; % 100 hz spiking

s.connections(end+1).direction='R->X';
s.connections(end).mechanism_list={'synDoubleExp'};
s.connections(end).parameters={'gSYN',0.18, 'tauR',0.3, 'tauD',1.5, 'netcon', eye(nCells)}; 

s.connections(end+1).direction='Inh->R';
s.connections(end).mechanism_list={'synDoubleExp_variablegSYN'};
s.connections(end).parameters={'tauR',0.3,'tauD',1.5,'ESYN',-80}; 

s.connections(end+1).direction='Exc->R';
s.connections(end).mechanism_list={'synDoubleExp'};
s.connections(end).parameters={'gSYN',0.18, 'tauR',0.3, 'tauD',1.5, 'netcon', eye(nCells)}; 

s.connections(end+1).direction='X->R';
s.connections(end).mechanism_list={'synDoubleExp'};
s.connections(end).parameters={'gSYN',0.12, 'tauR',0.3, 'tauD',1.5, 'netcon',XRnetcon, 'ESYN',-80}; 

s.connections(end+1).direction='R->C';
s.connections(end).mechanism_list={'synDoubleExp_variablegSYN'};
s.connections(end).parameters={'netcon',ones(nCells,1)};

%% vary params
vary = cell(length(varies),3);
for i = 1:length(varies)
    vary{i,1} = varies(i).conxn;
    vary{i,2} = varies(i).param;
    vary{i,3} = varies(i).range;
end

%% simulate
tic;

simdata = dsSimulate(s,'tspan',[dt time_end], 'solver',solverType, 'dt',dt,...
  'downsample_factor',1, 'save_data_flag',0, 'save_results_flag',1,...
  'study_dir',study_dir, 'vary',vary, 'debug_flag', 0, 'verbose_flag',0,...
  'restricts',restricts);

simdata = rmfield(simdata,{'Exc_V','Inh_V','R_V','labels','simulator_options','model'});

% save(fullfile(study_dir,'simulation_results.mat'),'simdata','-v7.3');

toc;

end