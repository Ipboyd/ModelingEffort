function [simdata,s] = birdNetwork(study_dir,varies,netcons,options)
% network for bird IC parameters
% works for mouse parameters as well. Maybe I should rename this to
% "spiking network".
% 
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
dt = 0.1; %ms
time_end = options.time_end;

%% neuron populations
% tonic = bias = cells spontaneous firing

nCells = 4;
s = struct();

s.populations(1).name = 'Exc';
s.populations(1).equations = 'noconLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'t_ref',0.2}; % same length as pulses in Exc->Exc

s.populations(end+1).name='S';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/100};

s.populations(end+1).name='N';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;

s.populations(end+1).name = 'X';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/275};

s.populations(end+1).name='R';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_ad_inc',0.0005,'tau_ad',80};

s.populations(end+1).name='C';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = 1;
% s.populations(end).parameters = {'Ad_tau',0};

% % % Imask vector specifies the channels that receive Iapp
% Imask = ones(1,nCells);
% % Imask(3) = 0;
% s.populations(end+1).name='TD';
% s.populations(end).equations = 'LIF_Iapp';
% s.populations(end).size = nCells;
% s.populations(end).parameters = {'Itonic',7,'noise',0,'Imask',Imask,'toff',time_end};

%% connections
if ~isfield(netcons,'xrNetcon'), netcons.xrNetcon = zeros(nCells); end  %X->R cross-channel inhib
if ~isfield(netcons,'rcNetcon'), netcons.rcNetcon = ones(nCells,1); end %converge to cortical unit
if ~isfield(netcons,'rxNetcon'), netcons.rxNetcon = ones(nCells,1); end

XRnetcon = netcons.xrNetcon;
rcNetcon = netcons.rcNetcon;
rxNetcon = netcons.rxNetcon;  % input to cross-channel inhibition

% ms
epsc_rise = 0.4;
epsc_fall = 2;

s.connections(1).direction='Exc->Exc';
s.connections(1).mechanism_list={'IC'};
s.connections(1).parameters={'g_postIC',0.285,'label','E','ICdir',options.ICdir,'locNum',options.locNum}; % 100 hz spiking

s.connections(end+1).direction='Exc->R';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.023, 'tauR',epsc_rise, 'tauD',epsc_fall};

s.connections(end+1).direction='Exc->S';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.02, 'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0,'tauP',45,'delay',2}; %gsyn = 0.0185 for 1:1 spiking
% s.connections(end).mechanism_list={'dummy'};
% s.connections(end).parameters={'gSYN',0.15};

% noise cell
s.connections(end+1).direction = 'N->N';
s.connections(end).mechanism_list={'iNoise'};
s.connections(end).parameters={'gSYN',0.27};

% input noise cell to relay
s.connections(end+1).direction = 'N->R';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.0165, 'tauR',epsc_rise, 'tauD',epsc_fall, 'netcon',eye(nCells)}; 

% CaT-type current in R channels
s.connections(end+1).direction = 'R->R';
s.connections(end).mechanism_list={'iT'};
s.connections(end).parameters={'gSYN',0.005,'ESYN',-40}; 

% R channels
s.connections(end+1).direction = 'R->R';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.004,'ESYN',0,'netcon',[0 1 0 0;1 0 1 0;....
    0 1 0 1;0 0 1 0]};

% feed-forward inhibition
s.connections(end+1).direction = 'S->R';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.0165, 'tauR',3, 'tauD',15,'ESYN',-80,'delay',3,'fP',0.2,'tauP',500}; 

% recurrent inhibition via R->S
s.connections(end+1).direction = 'R->S';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.01, 'tauR',epsc_rise, 'tauD',epsc_fall,'ESYN',0,'fP',0,'tauP',500}; 

s.connections(end+1).direction='R->X';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.004, 'tauR',epsc_rise, 'tauD',epsc_fall,'fF',0.05,'delay',5,'tauF',500}; 

s.connections(end+1).direction='X->R';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.006, 'tauR',8, 'tauD',24, 'netcon',XRnetcon, 'ESYN',-80}; 

% X->S best explains why PV neurons mostly show suppressed response (Kanold)
s.connections(end+1).direction='X->S';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008, 'tauR',8, 'tauD',24, 'netcon',eye(nCells), 'ESYN',-80}; 

s.connections(end+1).direction='R->C';
s.connections(end).mechanism_list={'iPSC_C'};
s.connections(end).parameters={'tauR',epsc_rise, 'tauD',epsc_fall, 'netcon',rcNetcon};

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
  'parfor_flag',options.parfor_flag);

toc;

end