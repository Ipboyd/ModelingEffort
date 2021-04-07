% 1. plot performance grids and netcons from the target network netcon
% 2. run network with learned weights, plot performance grids and weights
% make sure that in genTrainingData, the following is not overwritten:
% - subz
% - ICdir
% - padToTime

cd('C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid')
dynasimPath = 'C:\Users\Kenny\Desktop\GitHub\DynaSim';
addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = 'training 001 birdTuning';

% load DNN data
trainingSet = 3;
snnDataDir = 'C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\data_recordings\DNN_results';
load(sprintf('%s\\training_set_%02i.mat',snnDataDir,trainingSet)); %need ICdir, output_training

ICfiles = dir([ICdir filesep '*.mat']);
data = perf_data;
subz = 1:24;

% plot performance grid and netcons
options.subPops = {'C'};
plotPerformanceGrids_new;

subplot(1,3,2); imagesc(netcons.xrNetcon);
colorbar; caxis([0 1])

subplot(1,3,3); imagesc(netcons.rcNetcon);
colorbar; caxis([0 1])


%% run Network with imported parameters
dnnDataDir = 'C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\data_recordings\DNN_results';
load(sprintf('%s\\learnedWeights_set%02i.mat',dnnDataDir,trainingSet)) % learned weights
netcons.xrNetcon = learned.IRnetcon;
netcons.rcNetcon = learned.RCnetcon';
netcons.xrNetcon = zeros(4);
netcons.xrNetcon(3,4) = 1;
netcons.rcNetcon = [1 1 1 1]';

padToTime = 3200;

genTrainingData;

%% plot scatter
figure;
scatter(output_training,[snn_spks.C.smoothed.song{1}, snn_spks.C.smoothed.song{2}])
r = corrcoef(output_training,[snn_spks.C.smoothed.song{1}, snn_spks.C.smoothed.song{2}]);
title(['r = ' num2str(r(1,2))]);
xlabel('training output (target)')
ylabel('SNN output w/ learned weights')
axis tight

