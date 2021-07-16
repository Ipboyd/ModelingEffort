% generate training data from custom-designed AIM network for optimizing
% network weights with matlab's DNN toolbox.
%
% inputs: user specified, raw IC output
% network structure: user specified

% cd('/Users/jionocon/Documents/MATLAB/MouseSpatialGrid')
dynasimPath = '../DynaSim';
% ICdir = 'ICSimStim/bird/full_grids/BW_0.004 BTM_3.8 t0_0.1 phase0.4900/s60_STRFgain1.00_20210104-221956';
% ICdir = 'ICSimStim/bird/full_grids/BW_0.004 BTM_3.8 t0_0.1 phase0.4900/s50_STRFgain1.00_20210104-114659';
% ICdir = 'ICSimStim/bird/full_grids/BW_0.004 BTM_3.8 t0_0.1 phase0.4900/s30_STRFgain1.00_20210104-165447';
% ICdir = 'ICSimStim/bird/full_grids/BW_0.004 BTM_3.8 t0_0.1 phase0.4900/s20_STRFgain1.00_20210106-133343';
% ICdir = 'ICSimStim/bird/full_grids/BW_0.004 BTM_3.8 t0_0.1 phase0.4900/s7_STRFgain1.00_20210107-173527';
% ICdir = 'ICSimStim/mouse/full_grids/BW_0.009 BTM_3.8 t0_0.1 phase0.499/s1.5_STRFgain0.50_20200514-181040';
ICdir = uigetdir('ICSimStim');

addpath(cd)
addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = 'training 001 mouseTuning';
addpath('cSPIKE'); InitializecSPIKE;

debug_flag = 0;
save_flag = 0;

% setup directory for current simulation
datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);
if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));
simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

% get indices of STRFS, all locations, excitatory inputs only
ICfiles = dir([ICdir filesep '*.mat']);
subz = 1:length(ICfiles);
% subz = [[20:-5:5],fliplr([6:9,11:14,16:19,21:24])]; %to match experiment data
% subz = [20:-5:5];
% subz = find(~contains({ICfiles.name},'s0')); % exclude masker-only.
% subz = find(contains({ICfiles.name},'s1m2'));
% subz = [1:4,5,10,15,20,6,12,18,24]; %single channel
% subz = [5,7,10,11]; %channels 1 & 2
fprintf('found %i files matching subz criteria/n',length(subz));

% check IC inputs
if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

%% define network parameters
clear varies

dt = 0.1; %ms

gsyn_same = 0.35;

% custom parameters
varies(1).conxn = '(Inh->Inh,Exc->Exc)';
varies(1).param = 'trial';
varies(1).range = 1:20;

% deactivate TD neuron
varies(end+1).conxn = 'TD';
varies(end).param = 'Itonic';
varies(end).range = 0;

% inh neuron = sharpen
varies(end+1).conxn = 'Inh->Inh';
varies(end).param = 'g_postIC';
varies(end).range = 0.18;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'gSYN';
varies(end).range = 0;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'delay';
varies(end).range = 3;

varies(end+1).conxn = 'Exc->Exc';
varies(end).param = 'g_postIC';
varies(end).range = 1;

varies(end+1).conxn = 'Exc->R';
varies(end).param = 'gSYN';
varies(end).range = gsyn_same;

varies(end+1).conxn = 'Exc->X';
varies(end).param = 'gSYN';
varies(end).range = gsyn_same;

varies(end+1).conxn = 'R';
varies(end).param = 'noise';
varies(end).range = 0.;

varies(end+1).conxn = 'X';
varies(end).param = 'noise';
% varies(end).range = [1.5];
varies(end).range = 0;

varies(end+1).conxn = 'X->R';
varies(end).param = 'gSYN';
varies(end).range = 0.35;

varies(end+1).conxn = 'C';
varies(end).param = 'noise';
varies(end).range = [0.0];

% R-C weights 0.18 by default
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN1';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN2';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN3';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN4';
varies(end).range = gsyn_same;

% display parameters
network_params = [{varies.conxn}' {varies.param}' {varies.range}']

% find varied parameter other, than the trials
varied_param = find(cellfun(@length,{varies.range})>1);
if length(varied_param) > 1
    varied_param = varied_param(2); 
else
    varied_param = 2;
end
expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');
numVaried = length(varies(varied_param).range);

% specify netcons
if debug_flag
    netcons.xrNetcon = zeros(4); % cross channel inhibition
    netcons.xrNetcon(2,1) = 1;
    % netcons.xrNetcon(1,4) = 1;
    % netcons.xrNetcon(4,1) = 1;
    % netcons.xrNetcon(4,2) = 1;
    netcons.rcNetcon = [1 1 1 1]';
end
%%% use runGenTrainingData to call specific trainingSets %%%
% for trainingSetNum = 2

netcons.irNetcon = zeros(4); %inh -> R; sharpening
netcons.tdxNetcon = zeros(4); % I2 -> I
netcons.tdrNetcon = zeros(4); % I2 -> R
%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz)); %ms
padToTime = 3200; %ms
label = {'E','I'};
for ICtype = [0,1] %only E no I
    % divide all times by dt to upsample the time axis
    spks = [];
    for z = 1:length(subz)
        disp(ICfiles(subz(z)+0).name); %read in E spikes only
        load([ICdir filesep ICfiles(subz(z)).name],'t_spiketimes');
        
        % convert spike times to spike trains. This method results in
        % dt = 1 ms
        temp = cellfun(@max,t_spiketimes,'UniformOutput',false);
        tmax = max([temp{:}])/dt;
        singleConfigSpks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
        for j = 1:size(t_spiketimes,1) %trials [1:10]
            for k = 1:size(t_spiketimes,2) %neurons [(1:4),(1:4)]
                if k < 5 %song 1
                    singleConfigSpks(j,k,round(t_spiketimes{j,k}/dt)) = 1;
                else
                    singleConfigSpks(j+10,k-4,round(t_spiketimes{j,k}/dt)) = 1;
                end
            end
        end
%         singleConfigSpks(:,3,:) = 0; % zero out the U channel
        
        trialStartTimes(z) = padToTime;
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize)); 
        end
        % concatenate
        spks = cat(3,spks,singleConfigSpks);
    end
    save(fullfile(study_dir, 'solve',sprintf('IC_spks_%s.mat',label{ICtype+1})),'spks');
end

% figure;
% plotSpikeRasterFs(logical(squeeze(spks(:,1,:))),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('IC spike')
% xlim([0 padToTime/dt])

%% run simulation
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_202');
options.plotRasters = 0;
options.time_end = size(spks,3)*dt; %ms;
options.locNum = [];
options.parfor_flag = 1;
[snn_out,s] = birdNetwork(study_dir,varies,netcons,options);

%% post-process

% calculate performance
data = struct();
dataOld = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
options.plotRasters = 0;
options.subPops = {'Exc','R','C'}; %individually specify population performances to plot
configName = cellfun(@(x) strsplit(x,'_'),{ICfiles(subz).name}','UniformOutput',false);
configName = vertcat(configName{:});
configName = configName(:,1);
options.variedField = strrep(expVar,'-','_');
tic
for z = 1:length(subz)
    trialStart = PPtrialStartTimes(z);
    trialEnd = PPtrialEndTimes(z);
    figName = [simDataDir filesep configName{z}];
    [data(z).perf,data(z).fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    
%     [dataOld(z).perf,dataOld(z).fr] = postProcessData(snn_out,trialStart,trialEnd,options);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results
figure;
vizNetwork(s,0,'C','Exc')

% data = dataOld;
% plotPerformanceGrids;
% 
targetIdx = find(cellfun(@(x) contains(x,'m0'),{ICfiles.name}));
mixedIdx = find(cellfun(@(x) ~contains(x,'m0') && ~contains(x,'s0'),{ICfiles.name}));

simOptions = struct;

simOptions.subz = subz;
simOptions.varies = varies;
simOptions.varied_param = varied_param;
simOptions.locationLabels = {'90','45','0','-90'};
simOptions.expVar = expVar;
simOptions.chanLabels = simOptions.locationLabels;

if length(subz) == 24
    options.subPops = {'C'};
    plotPerformanceGrids_v3(data,s,options.subPops,targetIdx,mixedIdx,simOptions);
    
%     subplot(1,3,2); imagesc(netcons.xrNetcon);
%     colorbar; caxis([0 1])
%     
%     subplot(1,3,3); imagesc(netcons.rcNetcon);
%     colorbar; caxis([0 1])
end

