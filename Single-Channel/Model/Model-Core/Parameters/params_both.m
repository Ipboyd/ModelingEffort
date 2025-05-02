%% Options struct

options = struct;
options.nCells = 1;
options.opto = 0;

if options.opto, nSims = 5; else, nSims = 1; end

options.mex_flag = 0;
options.parfor_flag = 0;
options.plotRasters = 0;

% locNum should be empty for full grids
options.locNum = 15;
options.SpatialAttention = 0;

%% define network parameters
clear varies

trialInds = repmat(1:20,nSims,1);

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = '(On->On,Off->Off)';
varies(1).param = 'trial';
varies(1).range =  trialInds(:)';
% % % DO NOT CHANGE THIS % % %

% Input strength
varies(end+1).conxn = '(On->On,Off->Off)';
varies(end).param = 'g_postIC';
varies(end).range = 0.11;
% E->E connections

%Vary the synaptic depression (fp) between E->E conncections
%varies(end+1).conxn = '(On->R1On,R1On->R2On,Off->R1Off,R1Off->R2Off)';
varies(end+1).conxn = '(Off->R1On,On->R1On,R1On->R2On)';
varies(end).param = 'fP';
varies(end).range = 0.1;

%Gsyn
%varies(end+1).conxn = '(On->R1On,R1On->R2On,Off->R1Off,R1Off->R2Off)';
varies(end+1).conxn = '(Off->R1On,R1On->R2On,On->R1On)';
varies(end).param = 'gSYN';
varies(end).range = 0.02;

%PVs
varies(end+1).conxn = '(S1OnOff->R1On,S2OnOff->R2On,S2OnOff->R2Off)';
varies(end).param = 'gSYN';
varies(end).range = 0.025;

varies(end+1).conxn = '(S1OnOff->R1On,S2OnOff->R2On,S2OnOff->R2Off)';
varies(end).param = 'fP';
varies(end).range = 0.5;


% On -> PV
varies(end+1).conxn = '(On->S1OnOff,R1On->S2OnOff)';
varies(end).param = 'gSYN';
varies(end).range = 0.010;
%varies(end).range = [0.045:0.005:0.09];
%varies(end).range = [0.005:0.005:0.05];

% Off-> PV
varies(end+1).conxn = '(Off->S1OnOff)';
varies(end).param = 'gSYN';
varies(end).range = 0.030;
%varies(end).range = [0.005:0.005:0.05];
%varies(end).range = [0.075:0.005:0.12];

%varies(end+1).conxn = '(On->S1OnOff,R1On->S2OnOff)';
varies(end+1).conxn = '(On->S1OnOff)';
varies(end).param = 'fP';
varies(end).range = 0.2;

%varies(end+1).conxn = '(Off->S1OnOff,R1Off->S2OnOff)';
varies(end+1).conxn = '(Off->S1OnOff,R1On->S2OnOff)';
varies(end).param = 'fP';
varies(end).range = 0;


% control and opto conditions 
varies(end+1).conxn = '(S1OnOff,S2OnOff)';
varies(end).param = 'Itonic';
varies(end).range = 0; 

varies(end+1).conxn = '(R2On->R2On)';
varies(end).param = 'FR';
varies(end).range = 8;

% find varied parameter, excluding trials
varied_param = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));
if numel(varies(1).range) == 20, varied_param(1) = []; % delete trial varies
    end

if isempty(varied_param) % if no varied params, settle on 2nd entry in varies
    varied_param = 2;
end

% netcons can't be put in the varies struct (Dynasim doesn't recognize it?);
% it needs to be put in a different struct

netcons = struct; % row = source, column = target
netcons.XRnetcon = zeros(options.nCells,options.nCells);

% for simplification: use 1st varied param for 2d searches
expVar = [varies(varied_param(1)).conxn '-' varies(varied_param(1)).param];
expVar = strrep(expVar,'->','_');