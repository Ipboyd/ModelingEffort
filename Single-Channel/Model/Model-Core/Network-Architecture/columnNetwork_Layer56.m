function [simdata,s] = columnNetwork_paper(study_dir,varies,options,flag_raised_mex)

% Generates and simulates a network featuring columns of excitatory cells 
% that respond to onsets and offsets in auditory stimuli

% Features of circuit:

% Separate PV cells for onsets and offsets in stimuli; the PV cells will
% inhibit both onset and offset relay units in the same layer

% This is the original version of the single-channel model published in
% bioRxiv (Nocon et al 2022)

% INPUTS:
% study_dir: location of IC spike files + directory for log and data files
% varies: vary parameter, a structure. e.g.
%   varies(1).conxn = '(IC->IC)';
%   varies(1).param = 'trial';
%   varies(1).range = 1:20;
%   *first set of parameters should always be "trial" for IC->IC cxn
% options: struct with fields
%   nCells: number of channels in network
%   opto: simulating optogenetic suppression
%   SpatialAttention: =1 if simulating SpatialAttention
%   dt: timestep of trial (in ms)
%   time_end: length of simulations

% OUTPUTS:
% simdata - simulated voltages and spikes for all units
% s - model architecture (populations and connections)

%% Input check
if ~strcmp(varies(1).param,'trial')
    error('first set of varied params should be ''trial''')
end

%% solver params
solverType = 'euler';
dt = options.dt; %ms
time_end = options.time_end;

%% neuron populations
% tonic = bias = cells spontaneous firing

nCells = options.nCells;

s = struct();

% onset column

s.populations(1).name = 'On';
s.populations(1).equations = 'noconLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'t_ref',1};

s.populations(end+1).name = 'Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'t_ref',1};

s.populations(end+1).name='R1On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='R1Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='S1OnOff';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_L',1/100,'E_L',-57,'V_reset',-52,'t_ref',0.5};

s.populations(end+1).name='R2On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='R2Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='S2OnOff';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_L',1/100,'E_L',-57,'V_reset',-52,'t_ref',0.5};

s.populations(end+1).name='R3On';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='R3Off';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_inc',0.0003,'tau_ad',100,'t_ref',1};

s.populations(end+1).name='S3OnOff';
s.populations(end).equations = 'noconLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'g_L',1/100,'E_L',-57,'V_reset',-52,'t_ref',0.5};

%% connections

% ms
EE_rise = 0.7;  EE_fall = 1.5;   % E->E
IE_rise = 1;    IE_fall = 4.5;   % PV->E
EI_rise = 0.1;  EI_fall = 1;     % E->PV
XE_rise = 2.5;  XE_fall = 6;     % SOM->E

% note: all rise times must be greater than dt
if any([EE_rise,IE_rise,EI_rise,XE_rise,EE_fall,IE_fall,EI_fall,XE_fall]) <= dt
    error('PSC time constants must be greater than simulation timestep.')
end

%Changed from 0.265 3/25/2024 for RI performance criteria.
s.connections(1).direction='On->On';
s.connections(1).mechanism_list={'IC'};
s.connections(1).parameters={'g_postIC',0.19,'label','on','trial',1,'locNum',options.locNum,'netcon',eye(nCells,nCells),'t_ref',1,'t_ref_rel',1,'rec',2};

s.connections(end+1).direction='Off->Off';
s.connections(end).mechanism_list={'IC'};
s.connections(end).parameters={'g_postIC',0.19,'label','off','trial',1,'locNum',options.locNum,'netcon',eye(nCells,nCells),'t_ref',1,'t_ref_rel',1,'rec',2};

% % % L4 % % %

%Looking at E->PV fp = 0

% excitatory inputs
s.connections(end+1).direction='On->R1On';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

s.connections(end+1).direction='On->S1OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80};

s.connections(end+1).direction='S1OnOff->R1On';
s.connections(end).mechanism_list={'PSC'};
%s.connections(end).parameters={'gSYN',0.03,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 
%Pre Optimization
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

s.connections(end+1).direction='S1OnOff->R1Off';
s.connections(end).mechanism_list={'PSC'};
%s.connections(end).parameters={'gSYN',0.03,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120};
%Pre Optimization
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

% offset channels

s.connections(end+1).direction='Off->R1Off';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};


s.connections(end+1).direction='Off->S1OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0,'tauP',80};


% % %  L2/3  % % %

% excitatory inputs
s.connections(end+1).direction='R1On->R2On';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

s.connections(end+1).direction='R1On->S2OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80};

s.connections(end+1).direction='S2OnOff->R2On';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

s.connections(end+1).direction='S2OnOff->R2Off';
s.connections(end).mechanism_list={'PSC'};
%s.connections(end).parameters={'gSYN',0.01,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 
%Pre Optimization
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

% offset channels
s.connections(end+1).direction='R1Off->R2Off';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

s.connections(end+1).direction='R1Off->S2OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0,'tauP',80};


% % %  L5/6  % % %

% excitatory inputs
s.connections(end+1).direction='R2On->R3On';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

s.connections(end+1).direction='R2On->S3OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0.2,'tauP',80};

s.connections(end+1).direction='S3OnOff->R3On';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

s.connections(end+1).direction='S3OnOff->R3Off';
s.connections(end).mechanism_list={'PSC'};
%s.connections(end).parameters={'gSYN',0.01,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 
%Pre Optimization
s.connections(end).parameters={'gSYN',0.025,'tauR',IE_rise,'tauD',IE_fall,'ESYN',-80,'fP',0.5,'tauP',120}; 

% offset channels
s.connections(end+1).direction='R2Off->R3Off';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

s.connections(end+1).direction='R2Off->S3OnOff';
s.connections(end).mechanism_list={'PSC'};
s.connections(end).parameters={'gSYN',0.03,'tauR',EI_rise,'tauD',EI_fall,'fP',0,'tauP',80};

s.connections(end+1).direction='R3On->R3On';
s.connections(end).mechanism_list={'iNoise_V3'};
s.connections(end).parameters={'nSYN',0.015,'tauR_N',EE_rise,'tauD_N',EE_fall,'simlen',options.time_end / dt}; 

%Denser Inputs
%Todo Create condition for these (Comment out when not in use)
% s.connections(end+1).direction='Off->R1On';
% s.connections(end).mechanism_list={'PSC'};
% s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};
% 
% s.connections(end+1).direction='On->R1Off';
% s.connections(end).mechanism_list={'PSC'};
% s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};
% 
% s.connections(end+1).direction='R1On->R2Off';
% s.connections(end).mechanism_list={'PSC'};
% s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};
% 
% s.connections(end+1).direction='R1Off->R2On';
% s.connections(end).mechanism_list={'PSC'};
% s.connections(end).parameters={'gSYN',0.02,'tauR',EE_rise,'tauD',EE_fall,'fP',0.1,'tauP',30};

%% vary params
vary = cell(length(varies),3);
for i = 1:length(varies)
    vary{i,1} = varies(i).conxn;
    vary{i,2} = varies(i).param;
    vary{i,3} = varies(i).range;
end

% do dsVary2Modifications here to save time and reduce # of opto
% simulations
I_ind = find(strcmp(vary(:,2),'Itonic')); FR_ind = find(matches(vary(:,2),'FR'));

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

netcons = struct; 

% PEnetcon: PV->E, model as Gaussians for now
sigma = 30;
%netcons.PEnetcon = makePENetcon(bestLocs,sigma);
netcons.PEnetcon = eye(1);
netcons.XRnetcon = eye(1);
netcons.RCnetcon = eye(1);

% % poolobj = parpool('local', 8);
% if ~flag_raised_mex
%    simdata = dsSimulate(s, netcons, 'tspan',[dt time_end], 'solver',solverType, 'dt',dt,...
%   'downsample_factor',1, 'save_data_flag',0, 'save_results_flag',1,...
%   'study_dir',study_dir, 'vary',vary, 'debug_flag', 1, 'verbose_flag',0, ...
%   'parfor_flag',0, 'compile_flag',1);
%    copyfile('run\1-channel-paper\solve\solve_ode_1_channel_paper.m','mexes')
%    copyfile('run\1-channel-paper\solve\solve_ode_1_channel_paper_mex.mexw64','mexes')
% 
%     study_dir = fullfile(pwd,'run','1-channel-paper');
%     % 
%     % if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
%     % mkdir(fullfile(study_dir, 'solve'));
% 
%     solve_directory = fullfile(study_dir, 'solve');
%     mfileinfo = mfilename('fullpath');
% 
%     warning('off','all');
%     mkdir('backup');
% 
%     mfiledir = strsplit(mfileinfo,filesep);
% 
%     copyfile('run\1-channel-paper\solve\params.mat', 'backup');
% 
%     %if exist(fullfile(study_dir, 'solve'), 'dir')
%         % don't remove the directory
%     %else
%         %if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
%         %mkdir(solve_directory); 
%         flag_raised_mex = 0;
% 
%         mexes_dir = fullfile(mfiledir{1:end-1}, 'mexes');
%         if isfolder(mexes_dir)
%             m_file_to_copy = 'solve_ode_1_channel_paper.m';
%             mex_file_to_copy = 'solve_ode_1_channel_paper_mex.mexw64';
%             mex_file_path = fullfile(mexes_dir, mex_file_to_copy);
%             mex_files = dir([mex_file_path, '.*']);
%             if ~isempty(mex_files)
%                 flag_raised_mex = 1;
%                 for num = 1:20
%                     simDir = fullfile(solve_directory, ['sim' num2str(num)]);
%                     mkdir(simDir);
%                     copyfile(fullfile(mexes_dir, mex_files.name), simDir);
%                     copyfile(fullfile(mexes_dir, m_file_to_copy), simDir);
%                     copyfile('backup\params.mat', simDir);
%                 end
%             end
%         end
% 
% 
% 
% end


simdata = dsSimulate(s, netcons, 'tspan',[dt time_end], 'solver',solverType, 'dt',dt,...
  'downsample_factor',1, 'save_data_flag',0, 'save_results_flag',1,...
  'study_dir',study_dir, 'vary',vary, 'debug_flag', 1, 'verbose_flag',0, ...
  'parfor_flag',0, 'compile_flag',0);

end