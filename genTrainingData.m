% generate training data from custom-designed AIM network for optimizing
% network weights with matlab's DNN toolbox.
%
% inputs: user specified, raw IC output
% network structure: user specified

cd('C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid')
dynasimPath = 'C:\Users\Kenny\Desktop\GitHub\DynaSim';
% ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s60_STRFgain1.00_20210104-221956';
% ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s50_STRFgain1.00_20210104-114659';
% ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s30_STRFgain1.00_20210104-165447';
% ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s20_STRFgain1.00_20210106-133343';
ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s7_STRFgain1.00_20210107-173527';

addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = 'training 001 birdTuning';

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
% subz = find(~contains({ICfiles.name},'s0')); % exclude masker-only.
fprintf('found %i files matching subz criteria\n',length(subz));

% check IC inputs
if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

%% define network parameters
clear varies

dt = 0.1; %ms

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
varies(end).range = 0.25;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'gSYN';
varies(end).range = 0;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'delay';
varies(end).range = 3;

varies(end+1).conxn = 'Exc->Exc';
varies(end).param = 'g_postIC';
varies(end).range = [0.27,0.3];

varies(end+1).conxn = 'Exc->R';
varies(end).param = 'gSYN';
varies(end).range = 0.20;

varies(end+1).conxn = 'R';
varies(end).param = 'noise';
varies(end).range = 0.01;

varies(end+1).conxn = 'X';
varies(end).param = 'noise';
varies(end).range = [1.5];

varies(end+1).conxn = 'X->R';
varies(end).param = 'gSYN';
varies(end).range = [0.25];

varies(end+1).conxn = 'C';
varies(end).param = 'noise';
varies(end).range = [0.01];

% R-C weights 0.18 by default
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN1';
varies(end).range = 0.18;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN2';
varies(end).range = 0.18;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN3';
varies(end).range = 0.18;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN4';
varies(end).range = 0.18;

% display parameters
network_params = [{varies.conxn}' {varies.param}' {varies.range}']

% find varied parameter other, than the trials
varied_param = find(cellfun(@length,{varies.range})>1);
if length(varied_param) > 1
    varied_param = varied_param(2); 
else
    varied_param = 1;
end
expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');


% specify netcons
% netcons.xrNetcon = zeros(4); % cross channel inhibition
% netcons.xrNetcon(2,1) = 1;
% netcons.xrNetcon(3,1) = 1;
% netcons.xrNetcon(4,1) = 1;
% netcons.xrNetcon(2,4) = 1;

for trainingSetNum = 2

netcons.xrNetcon = zeros(4);
netcons.irNetcon = zeros(4); %inh -> R; sharpening
netcons.tdxNetcon = zeros(4); % I2 -> I
netcons.tdrNetcon = zeros(4); % I2 -> R
netcons.rcNetcon = [1 0 0 0]';
%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz));
padToTime = 2000; %ms
label = {'E','I'};
for ICtype = [0,1] %only E no I
    spks = [];
    for z = 1:length(subz)
        disp(ICfiles(subz(z)+0).name); %read in E spikes only
        load([ICdir filesep ICfiles(z).name],'t_spiketimes');
        
        % convert spike times to spike trains. This method results in
        % dt = 1 ms
        temp = cellfun(@max,t_spiketimes,'UniformOutput',false);
        tmax = max([temp{:}]);
        singleConfigSpks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
        for j = 1:size(t_spiketimes,1) %trials [1:10]
            for k = 1:size(t_spiketimes,2) %neurons [(1:4),(1:4)]
                if k < 5 %song 1
                    singleConfigSpks(j,k,round(t_spiketimes{j,k})) = 1;
                else
                    singleConfigSpks(j+10,k-4,round(t_spiketimes{j,k})) = 1;
                end
            end
        end
        
        trialStartTimes(z) = padToTime;
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime
            padSize = padToTime-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize)); 
        end
        % concatenate & upsample - pretty important
        spks = cat(3,spks,singleConfigSpks);
    end
    spks = imresizen(spks,[1,1,1/dt]);
    save(fullfile(study_dir, 'solve',sprintf('IC_spks_%s.mat',label{ICtype+1})),'spks');
end

%% run simulation
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_2020');
options.plotRasters = 0;
options.time_end = size(spks,3)*dt; %ms;
options.locNum = [];
options.parfor_flag = 1;
[temp,s] = birdNetwork(study_dir,varies,netcons,options);

%% post process

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
    [data(z).perf,data(z).fr] = postProcessData_new(temp,s,trialStart,trialEnd,figName,options);
    
%     [dataOld(z).perf,dataOld(z).fr] = postProcessData(temp,trialStart,trialEnd,options);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results
% figure;
% vizNetwork(s,0,'C','Exc')

% data = dataOld;
% plotPerformanceGrids;

% options.subPops = {'C'};
% plotPerformanceGrids_new;

%% Smooth input and output data
t = 0:0.001:0.100; % 0-100 ms, 1 ms interval
tau = 0.100; %100 ms
kernel = t.*exp(-t/tau);

% amount of delay between input and output, in units of taps
NumDelayTaps = 40;

% input data: change from [trials x neurons x configs] to [(configs x trials) x neurons]
input_spks = spks(:,:,1:end-NumDelayTaps); %account for delay
input_spks = [sum(input_spks(1:10,:,:)) ; sum(input_spks(11:end,:,:))]; %psth
structured_input = reshape(permute(input_spks,[3,1,2]),[],4);
smoothed_input = conv2(structured_input,kernel','same');

% output data
numVaried = 2;
Cspks = [temp(1:numVaried:end).C_V_spikes];
Cspks = Cspks(1+NumDelayTaps:end,:); %account for delay
figure;
plotSpikeRasterFs(logical(Cspks'),'PlotType','vertline2','Fs',1/dt);
xlim([0 temp(1).time(end)/dt])

Cspks = [sum(Cspks(:,1:10),2), sum(Cspks(:,11:20),2)];
structured_Cspks = Cspks(:);
smoothed_Cspks = conv(structured_Cspks,kernel,'same');


% figure;
% imagesc(smoothed_Cspks')

input_training = smoothed_input;
output_training = smoothed_Cspks;
perf_data = data;
% name = input('name of training set? ');
name = sprintf('training_set_%i',trainingSetNum);
save(['SNN_optimization\' name '.mat'],'input_training','output_training','perf_data','netcons','network_params','Cspks','options');
end