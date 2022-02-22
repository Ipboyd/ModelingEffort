function [simdata,s] = columnNetwork(study_dir,varies,netcons,options)
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


% @Jio Nocon, BU 2022-01-20
% Added L4 layer between TC and L2/3 (per Moore and Wehr paper)


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

nCells = 1;
s = struct();

s.populations(1).name = 'On';
s.populations(1).equations = 'noconLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'t_ref',0.2}; % same length as pulses in Exc->Exc

s.populations(end+1).name = 'Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'t_ref',0.2}; % same length as pulses in Exc->Exc

% onset column

s.populations(end+1).name='S1On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/100,'E_leak',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R1On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_ad_inc',0.00035,'tau_ad',60,'t_ref',1.5};

s.populations(end+1).name='S2On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/100,'E_leak',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R2On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_ad_inc',0.00035,'tau_ad',60,'t_ref',1.5};

% offset cells

s.populations(end+1).name='S1Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/100,'E_leak',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R1Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_ad_inc',0.00035,'tau_ad',60,'t_ref',1.5};

s.populations(end+1).name='S2Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_leak',1/100,'E_leak',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R2Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_ad_inc',0.00035,'tau_ad',60,'t_ref',1.5};

% convergence
s.populations(end+1).name='C';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = 1;
% s.populations(end).parameters = {'g_ad_inc',0.0005,'tau_ad',80};

%% connections
if ~isfield(netcons,'xrNetcon'), netcons.xrNetcon = zeros(nCells); end  % X->R cross-channel inhib
if ~isfield(netcons,'rcNetcon'), netcons.rcNetcon = ones(nCells,1); end % converge to cortical unit

rcNetcon = netcons.rcNetcon;

% ms
epsc_rise = 0.4;
epsc_fall = 2;

% onset and offset cell inputs
s.connections(1).direction='On->On';
s.connections(1).mechanism_list={'IC'};
s.connections(1).parameters={'g_postIC',0.265,'label','On','ICdir',options.ICdir,'locNum',options.locNum}; % 100 hz spiking

s.connections(end+1).direction='Off->Off';
s.connections(end).mechanism_list={'IC'};
s.connections(end).parameters={'g_postIC',0.265,'label','Off','ICdir',options.ICdir,'locNum',options.locNum}; % 100 hz spiking

% % % L4 % % %

% excitatory inputs
s.connections(end+1).direction='On->R1On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008, 'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.016,'tauP',80};

s.connections(end+1).direction='On->S1On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.012, 'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.04,'tauP',100}; %gsyn = 0.0165 for 1:1 spiking
 
% inputs to sharpening units
s.connections(end+1).direction='Off->R1Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008, 'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.016,'tauP',80};

s.connections(end+1).direction='Off->S1Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.012, 'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.04,'tauP',100}; %gsyn = 0.0165 for 1:1 spiking

% feed-forward inhibition in L4
s.connections(end+1).direction = 'S1On->R1On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008,'tauR',1.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',160}; 

 s.connections(end+1).direction = 'S1Off->R1Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008,'tauR',1.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',160}; 

% CaT-type current and Poisson noise in R channels
s.connections(end+1).direction = 'R1On->R1On';
s.connections(end).mechanism_list={'iT','iNoise_V2'};
s.connections(end).parameters={'gSYN',0.007,'ESYN',-40,'nSYN',0.015, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall,'locNum',options.locNum,'netcon',eye(nCells)}; 

s.connections(end+1).direction = 'R1Off->R1Off';
s.connections(end).mechanism_list={'iT','iNoise_V2'};
s.connections(end).parameters={'gSYN',0.007,'ESYN',-40,'nSYN',0.015, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall,'locNum',options.locNum,'netcon',eye(nCells)}; 

% Poisson noise at Pvalb
s.connections(end+1).direction = 'S1On->S1On';
s.connections(end).mechanism_list={'iNoise_V2'};
s.connections(end).parameters={'nSYN',0.016,'tauR_N',epsc_rise, 'tauD_N',epsc_fall, 'netcon',eye(nCells)}; 

s.connections(end+1).direction = 'S1Off->S1Off';
s.connections(end).mechanism_list={'iNoise_V2'};
s.connections(end).parameters={'nSYN',0.016,'tauR_N',epsc_rise, 'tauD_N',epsc_fall, 'netcon',eye(nCells)}; 

% % cross-column connections
% s.connections(end+1).direction = 'S1Off->R1On';
% s.connections(end).mechanism_list={'iPSC_LTP'};
% s.connections(end).parameters={'gSYN',0.007, 'tauR',2.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',200}; 
% 
% s.connections(end+1).direction = 'S1On->R1Off';
% s.connections(end).mechanism_list={'iPSC_LTP'};
% s.connections(end).parameters={'gSYN',0.007, 'tauR',2.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',200}; 

% % %  L2/3  % % %

s.connections(end+1).direction = 'R1On->R2On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.015,'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.016,'tauP',80};

s.connections(end+1).direction = 'R1On->S2On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.012,'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.04,'tauP',100}; %gsyn = 0.0165 for 1:1 spiking

% feed-forward inhibition in L2/3
s.connections(end+1).direction = 'S2On->R2On';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008,'tauR',1.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',160}; 

s.connections(end+1).direction = 'S2Off->R2Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.008,'tauR',1.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',160}; 

s.connections(end+1).direction = 'R1Off->R2Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.015,'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.016,'tauP',80};

s.connections(end+1).direction = 'R1Off->S2Off';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.012,'tauR',epsc_rise, 'tauD',epsc_fall,'fP',0.04,'tauP',100}; %gsyn = 0.0165 for 1:1 spiking

% CaT-type current and Poisson noise in R channels
s.connections(end+1).direction = 'R2On->R2On';
s.connections(end).mechanism_list={'iT','iNoise_V2'};
s.connections(end).parameters={'gSYN',0.007,'ESYN',-40,'nSYN',0.015, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall,'locNum',options.locNum,'netcon',eye(nCells)}; 

s.connections(end+1).direction = 'R2Off->R2Off';
s.connections(end).mechanism_list={'iT','iNoise_V2'};
s.connections(end).parameters={'gSYN',0.007,'ESYN',-40,'nSYN',0.015, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall,'locNum',options.locNum,'netcon',eye(nCells)}; 

% Poisson noise at Pvalb
s.connections(end+1).direction = 'S2On->S2On';
s.connections(end).mechanism_list={'iNoise_V2'};
s.connections(end).parameters={'nSYN',0.016, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall, 'netcon',eye(nCells)}; 

s.connections(end+1).direction = 'S2Off->S2Off';
s.connections(end).mechanism_list={'iNoise_V2'};
s.connections(end).parameters={'nSYN',0.016, 'tauR_N',epsc_rise, 'tauD_N',epsc_fall, 'netcon',eye(nCells)}; 

% % cross-column connections
% s.connections(end+1).direction = 'S2Off->R2On';
% s.connections(end).mechanism_list={'iPSC_LTP'};
% s.connections(end).parameters={'gSYN',0.007, 'tauR',2.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',200}; 
% 
% s.connections(end+1).direction = 'S2On->R2Off';
% s.connections(end).mechanism_list={'iPSC_LTP'};
% s.connections(end).parameters={'gSYN',0.007, 'tauR',2.5, 'tauD',5,'ESYN',-80,'delay',0,'fP',0.06,'tauP',200}; 

% convergence onto readout cell
s.connections(end+1).direction='R2On->C';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.009,'tauR',2, 'tauD',3, 'netcon',1};

s.connections(end+1).direction='R2Off->C';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0,'tauR',epsc_rise, 'tauD',epsc_fall, 'netcon',1};

s.connections(end+1).direction='S2On->C';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.02,'tauR',2, 'tauD',3,'ESYN',-80,'delay',0,'fP',0.2,'tauP',160}; 

s.connections(end+1).direction='S2Off->C';
s.connections(end).mechanism_list={'iPSC_LTP'};
s.connections(end).parameters={'gSYN',0.02,'tauR',2, 'tauD',3,'ESYN',-80,'delay',0,'fP',0.2,'tauP',160}; 

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