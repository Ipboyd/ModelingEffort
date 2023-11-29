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

% E->E connections
varies(end+1).conxn = '(On->R1On,R1On->R2On,Off->R1Off,R1Off->R2Off)';
varies(end).param = 'gSYN';
varies(end).range = 0.02;

% onset pvs
varies(end+1).conxn = '(S1On->R1On,S1On->R1Off,S2On->R2On,S2On->R2Off)';
varies(end).param = 'gSYN';
varies(end).range = 0.025*0.8;

% offset pvs
varies(end+1).conxn = '(S1Off->R1On,S1Off->R1Off,S2Off->R2On,S2Off->R2Off)';
varies(end).param = 'gSYN';
varies(end).range = 0.035*0.8;

% control and opto conditions 
varies(end+1).conxn = '(S1On,S1Off,S2On,S2Off)';
varies(end).param = 'Itonic';
varies(end).range = 0; 

varies(end+1).conxn = '(R2On->R2On,S2On->S2On,S2Off->S2Off)';
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