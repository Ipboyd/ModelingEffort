function [simdata,s] = columnNetwork(study_dir,varies,options,netcons,model)
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

nCells = options.nCells;

s = struct();

XRnetcon = netcons.XRnetcon;

% onset column

s.populations(1).name = 'IC';
s.populations(1).equations = 'noconLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'t_ref',0.2};

s.populations(end+1).name='S1';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_L',1/100,'E_L',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R1';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.00035,'tau_ad',60,'t_ref',1.5};

s.populations(end+1).name='S2';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_L',1/100,'E_L',-56,'V_reset',-50,'t_ref',0.8};

s.populations(end+1).name='R2';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.00035,'tau_ad',60,'t_ref',1.5};

s.populations(end+1).name='X1';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0002,'tau_ad',60,'g_L',1/275,'t_ref',1.5};

s.populations(end+1).name='X2';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0002,'tau_ad',60,'g_L',1/275,'t_ref',1.5};

% % VIP cells
% s.populations(end+1).name='T';
% s.populations(end).equations = 'noconLIF';
% s.populations(end).size = nCells;
% s.populations(end).parameters = {'t_ref',0,'V_reset',-65,'Itonic',9,'Imask',Imask};

% convergence
s.populations(end+1).name='C';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = 1;
s.populations(end).parameters = {'g_inc',0.00035,'tau_ad',60,'t_ref',1.5};

%% connections

% ms
EE_rise = 0.5; EE_fall = 3;
IE_rise = 1; IE_fall = 4;
EI_rise = 0.1; EI_fall = 1;
EX_rise = 1.5; EX_fall = 4;
XE_rise = 10;  XE_fall = 50;

% % % % % % % % onset column % % % % % % % % 

s.connections(1).direction='IC->IC';
s.connections(1).mechanism_list={'IC'};
s.connections(1).parameters={'g_postIC',0.265,'trial',1,'locNum',options.locNum,'netcon',eye(nCells,nCells)};

% % % L4 % % %

% excitatory inputs
s.connections(end+1).direction='IC->R1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.011,'tauR',EE_rise,'tauD',EE_fall};

s.connections(end+1).direction='IC->S1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80};

% SOM inhibition
s.connections(end+1).direction='R1->X1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.0025,'tauR',EX_rise,'tauD',EX_fall,'fF',0.1,'tauF',180};

s.connections(end+1).direction='X1->R1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.003,'tauR',XE_rise,'tauD',XE_fall,'fF',0.1,'tauF',180,'ESYN',-80,'netcon',XRnetcon}; 

% feed-forward inhibition in L4
s.connections(end+1).direction='S1->R1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',80}; 

% recurrent excitation
s.connections(end+1).direction = 'R1->S1';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80}; 

% % %  L2/3  % % %

s.connections(end+1).direction = 'R1->R2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.011,'tauR',EE_rise,'tauD',EE_fall};

s.connections(end+1).direction = 'R1->S2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80}; %gsyn = 0.025 for 1:1 spiking

% feed-forward inhibition in L2/3
s.connections(end+1).direction = 'S2->R2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',80}; 

% recurrent excitation
s.connections(end+1).direction = 'R2->S2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80}; 

% SOM inhibition
s.connections(end+1).direction='R2->X2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.0025,'tauR',EX_rise,'tauD',EX_fall,'fF',0.1,'tauF',180};

s.connections(end+1).direction='X2->R2';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.003,'tauR',XE_rise,'tauD',XE_fall,'fF',0.1,'tauF',180,'ESYN',-80,'netcon',XRnetcon}; 

% s.connections(end+1).direction='X2->S2';
% s.connections(end).mechanism_list={'PSC'};
% s.connections(end).parameters={'gSYN',0.0025,'tauR',XI_rise,'tauD',XI_fall,'fF',0.1,'tauF',180,'ESYN',-80,'netcon',XRnetcon}; 

s.connections(end+1).direction='R2->R2';
s.connections(end).mechanism_list={'iNoise_V3'};
s.connections(end).parameters={'nSYN',0.011,'tauR_N',EE_rise, 'tauD_N',EE_fall,'locNum',options.locNum}; 

% convergence onto readout cell
s.connections(end+1).direction='R2->C';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.011,'tauR',EE_rise,'tauD',EE_fall,'netcon',ones(nCells,1)};

s.connections(end+1).direction='S2->C';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.4,'tauP',120,'netcon',ones(nCells,1)}; 

%% vary params
vary = cell(length(varies),3);
for i = 1:length(varies)
    vary{i,1} = varies(i).conxn;
    vary{i,2} = varies(i).param;
    vary{i,3} = varies(i).range;
end

% do dsVary2Modifications here to save time and reduce # of opto
% simulations
I_ind = find(strcmp(vary(:,2),'Itonic')); FR_ind = find(strcmp(vary(:,2),'(FR,sigma)'));

if ~isempty(I_ind) && ~isempty(FR_ind)
if numel(vary{I_ind,3}) > 1 && size(vary{FR_ind,3},2) > 1
    numSets = numel(vary{I_ind,3})*size(vary{FR_ind,3},2);
    
    sqs = (1:numel(vary{I_ind,3})).^2;
    nonsqs = setdiff(1:numSets,sqs);
    
    vary = dsVary2Modifications(vary);
    
    temp = [];
    for n = nonsqs, temp = cat(2,temp,n:numSets:numel(vary)); end
    
    vary(temp) = [];
end
end

%% simulate
tic;

simdata = dsSimulate(s,'tspan',[dt time_end], 'solver',solverType, 'dt',dt,...
  'downsample_factor',1, 'save_data_flag',0, 'save_results_flag',1,...
  'study_dir',study_dir, 'vary',vary, 'debug_flag', 1, 'verbose_flag',0,...
  'mex_flag',options.mex_flag);

toc;

end