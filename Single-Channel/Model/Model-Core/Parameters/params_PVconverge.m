%% Options struct

options = struct;
options.nCells = 1;
options.opto = 0;

options.mex_flag = 0;
options.parfor_flag = 0;
options.plotRasters = 0;

% locations of speakers: [90 45 0 -90];

% locNum should be empty for full grids
options.locNum = 15;
options.SpatialAttention = 0;

%% define network parameters
clear varies

if options.opto, nSims = 5; else, nSims = 1; end

trialInds = repmat(1:20,nSims,1);

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = '(On->On,Off->Off)';
varies(1).param = 'trial';
varies(1).range =  trialInds(:)';
% % % DO NOT CHANGE THIS % % %

% vary the depression strength and recovery at inputs to PV
varies(end+1).conxn = '(On->S1,R1On->S2,Off->S1,R1Off->S2)';
varies(end).param = 'tauP';
varies(end).range = [ 20 : 20 : 100 ];

varies(end+1).conxn = '(On->S1,R1On->S2,Off->S1,R1Off->S2)';
varies(end).param = 'fP';
varies(end).range = [ 0.2 : 0.2 : 1 ];

% noise at intermediate neurons
varies(end+1).conxn = 'R2On->R2On';
varies(end).param = 'FR';
varies(end).range = 8/options.nCells;
