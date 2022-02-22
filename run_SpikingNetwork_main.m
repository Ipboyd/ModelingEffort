% generate training data from custom-designed AIM network for optimizing
% network weights with matlab's DNN toolbox.
%
% inputs: user specified, raw IC output
% network structure: user specified

cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid')
dynasimPath = '../DynaSim';
% ICdir = uigetdir('ICSimStim');

% % 1.5ms refractory period
%ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\full_grids\alpha_83.333 N1_5 N2_7\s38_STRFgain0.80_20210914-161309';

% 3ms refractory period
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\full_grids//alpha_83.333 N1_5 N2_7//s38_STRFgain0.80_20210922-153448';

% 3ms refractory period, lower gain

%ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\full_grids\alpha_83.333 N1_5 N2_7\s38_STRFgain0.40_20210922-162928';

% for grant
ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\full_grids//alpha_83.333 N1_5 N2_7//s38_STRFgain0.40_20210923-211301';

addpath(cd)
addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = 'T-type only';
addpath('cSPIKE'); InitializecSPIKE;
addpath('fixed-stimuli');

debug_flag = 1;
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

% gSYN = maximal synaptic conductance [uS]
% default value: 0.015 uS ensures 1:1 spike transfer
% double exponential function is scaled such that max value is 1 regardless
% of rise and decay times

dt = 0.1; %ms
tonic_FR = 6; %Hz
Vdiff = 5; R = 200; C = 0.1; E_leak = -65; V_th = -47;
Itonic = 0.04; %nA
% C*(Vdiff*(tonic_FR/1000) - (E_leak - V_th)/(R*C)) / 2, analytical solution from Dayan and Abbott, 2001

Vstd = 6; % standard deviation of voltage [mV]
Rnoise = Vstd*C; %nA

% custom parameters
varies(1).conxn = 'Exc->Exc';
varies(1).param = 'trial';
varies(1).range = 1:20;

varies(end+1).conxn = 'S->R';
varies(end).param = 'gSYN';
varies(end).range = 0;

varies(end+1).conxn = 'X->R';
varies(end).param = 'gSYN';
varies(end).range = 0.012;

varies(end+1).conxn = 'R';
varies(end).param = 'Itonic';
varies(end).range = Itonic;

varies(end+1).conxn = 'R';
varies(end).param = 'noise';
varies(end).range = Rnoise;

varies(end+1).conxn = 'R->C';
varies(end).param = '(gSYN1,gSYN2,gSYN3,gSYN4)';
varies(end).range = [0.008;0;0.012;0.008];

% display parameters
network_params = [{varies.conxn}' {varies.param}' {varies.range}'];

% find varied parameter other, than the trials
varied_param = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));

if length(varied_param) > 1
    varied_param = varied_param(2); 
else % if no varied params, settle on 2nd entry in varies
    varied_param = 2;
end

expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');
numVaried = length(varies(varied_param).range);

% specify netcons
netcons.rxNetcon = zeros(4);
if debug_flag
    netcons.xrNetcon = zeros(4); % cross channel inhibition
    netcons.xrNetcon(3,4) = 1;
    netcons.xrNetcon(3,1) = 1;
    netcons.xrNetcon(3,2) = 1;
    netcons.rcNetcon = [1 0 1 1]';
end
% for trainingSetNum = 2

netcons.rxNetcon = zeros(4,4);
netcons.rxNetcon(3,3) = 1; % R -> X; drive cross-channel

%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz)); %ms
padToTime = 3500; %ms
label = {'E','I'};
for ICtype = [0,1] % only E no I
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
        singleConfigSpks(:,2,:) = 0; % zero out the U channel
        
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
options.parfor_flag = 0;
[snn_out,s] = birdNetwork(study_dir,varies,netcons,options);

%% post-process

% calculate performance
data = struct();
dataOld = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
options.plotRasters = 1;
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
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results
figure;
vizNetwork(s,0,'C','Exc')
saveas(figure(1),fullfile('simData',expName,'netcons.png'));
saveas(figure(2),fullfile('simData',expName,'network.png'));
close all;

% data = dataOld;
% plotPerformanceGrids;

targetIdx = find(cellfun(@(x) contains(x,'m0'),{ICfiles.name}));
mixedIdx = find(cellfun(@(x) ~contains(x,'m0') && ~contains(x,'s0'),{ICfiles.name}));

simOptions = struct;

annotTable = createSimNotes(snn_out,expName);

simOptions.subz = subz;
simOptions.varies = varies;
simOptions.varied_param = varied_param;
simOptions.locationLabels = {'90','45','0','-90'};
simOptions.expVar = expVar;
simOptions.chanLabels = simOptions.locationLabels;

if length(subz) == 24
    options.subPops = {'C'};
    plotPerformanceGrids_v3(data,s,annotTable,options.subPops,targetIdx,mixedIdx,simOptions,expName);
    
%     subplot(1,3,2); imagesc(netcons.xrNetcon);
%     colorbar; caxis([0 1])
%     
%     subplot(1,3,3); imagesc(netcons.rcNetcon);
%     colorbar; caxis([0 1])
end


