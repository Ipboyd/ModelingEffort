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
% subz = 1:length(ICfiles);
% subz = find(~contains({ICfiles.name},'s0')); % exclude masker-only.
subz = find(contains({ICfiles.name},'s1m0'));
fprintf('found %i files matching subz criteria\n',length(subz));

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
varies(end).range = [0.18];

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


% specify netcons
% netcons.xrNetcon = zeros(4); % cross channel inhibition
% netcons.xrNetcon(2,1) = 1;
% netcons.xrNetcon(3,1) = 1;
% netcons.xrNetcon(4,1) = 1;
% netcons.xrNetcon(2,4) = 1;

%%% use runGenTrainingData to call specific trainingSets %%%
% for trainingSetNum = 2

% netcons.xrNetcon = zeros(4);
netcons.irNetcon = zeros(4); %inh -> R; sharpening
netcons.tdxNetcon = zeros(4); % I2 -> I
netcons.tdrNetcon = zeros(4); % I2 -> R
netcons.rcNetcon = [1 1 1 1]';
%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz)); %ms
padToTime = 2000; %ms
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
        
        trialStartTimes(z) = padToTime;
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize)); 
        end
        % concatenate & upsample - pretty important
        spks = cat(3,spks,singleConfigSpks);
    end
%     spks = imresizen(spks,[1,1,1/dt]);
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
% 
% options.subPops = {'C'};
% plotPerformanceGrids_new;

%% Smooth input and output data
t = (0:dt:500)/1000; % 0-500ms
tau = 0.001; % second
kernel = t.*exp(-t/tau);

% amount of delay between input and output, in units of taps
NumDelayTapsL0 = 1; %E
NumDelayTapsL1 = 9; %R,X
NumDelayTapsL2 = 17; %C

snn_spks = [];
snn_spks.IC.delay = NumDelayTapsL2; %not strictly necessary for now 
snn_spks.E.delay = NumDelayTapsL0;
snn_spks.R.delay = NumDelayTapsL1;
snn_spks.X.delay = NumDelayTapsL1;
snn_spks.C.delay = NumDelayTapsL2;

% output data
Cspks = [temp(1:numVaried:end).C_V_spikes];

% intermediate neurons
Rspks = [];
Xspks = [];
Espks = [];
for i = 1:20 %number of trials
    snn_spks.R.raw(i,:,:) = temp(i).R_V_spikes;
    snn_spks.X.raw(i,:,:) = temp(i).X_V_spikes;
    snn_spks.E.raw(i,:,:) = temp(i).Exc_V_spikes;
end

% combine across trials, delay, smooth with kernel, remove zero-padded length
n = padToTime/dt;
for songn = 1:2
    delay = snn_spks.IC.delay;
    m = n-delay;
    snn_spks.IC.psth.song{songn} = squeeze(sum(spks((1:10) + 10*(songn-1),:,:)));
    snn_spks.IC.psth.song{songn} = snn_spks.IC.psth.song{songn}(:,1:end-delay);
    delayedSpks = snn_spks.IC.psth.song{songn};
    snn_spks.IC.smoothed.song{songn} = conv2(delayedSpks',kernel');
    snn_spks.IC.smoothed.song{songn} =  snn_spks.IC.smoothed.song{songn}(1:m,:);
    
    for neuron = {'R','X','E'}
        delay = snn_spks.(neuron{1}).delay;
        m = n-delay;
        snn_spks.(neuron{1}).psth.song{songn} = squeeze(sum(snn_spks.(neuron{1}).raw((1:10) + 10*(songn-1),:,:)));
        snn_spks.(neuron{1}).psth.song{songn} = snn_spks.(neuron{1}).psth.song{songn}(1+delay:end,:);
        delayedSpks = snn_spks.(neuron{1}).psth.song{songn};
        snn_spks.(neuron{1}).smoothed.song{songn} = conv2(delayedSpks,kernel');
        snn_spks.(neuron{1}).smoothed.song{songn} =  snn_spks.(neuron{1}).smoothed.song{songn}(1:m,:);
    end
    
    delay = snn_spks.C.delay;
    m = n-delay;
    snn_spks.C.psth.song{songn} = sum(Cspks(:,(1:10) + 10*(songn-1)),2);
    snn_spks.C.psth.song{songn} = snn_spks.C.psth.song{songn}(1+delay:end,:);
    delayedSpks = snn_spks.C.psth.song{songn};
    snn_spks.C.smoothed.song{songn} = conv(delayedSpks',kernel);
    snn_spks.C.smoothed.song{songn} = snn_spks.C.smoothed.song{songn}(1:m);
end

% visualizations
channel = 1;
for songn = 1:2
    figure;
    for neuron = {'IC','E','R','X'}
        current_psth = snn_spks.(neuron{1}).smoothed.song{songn}(:,channel);
        plot(current_psth,'linewidth',1.5,'linestyle','-'); hold on;
    end
    plot(snn_spks.C.smoothed.song{songn},'linewidth',1.5,'linestyle','-.');
    xlim([2500 3200])
    legend('in','E','R','X','out')
end


%%%%%%%%%%%%%%%%%%%%%% save results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_training = smoothed_input;
output_training = smoothed_Cspks;
perf_data = data;
% name = input('name of training set? ');
% name = sprintf('training_set_%i',trainingSetNum);
% save(['SNN_optimization' filesep name '.mat'],'input_training','output_training',...
%     'perf_data','netcons','network_params','Cspks','options',...
%     'Rspks','smoothed_R','Xspks','smoothed_X');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end

%%
% figure;
% plotSpikeRasterFs(logical(Espks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 temp(1).time(end)/dt])
% title('E spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(Rspks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 temp(1).time(end)/dt])
% title('R spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(squeeze(spks(:,1,:))),'PlotType','vertline2','Fs',1/dt);
% xlim([0 temp(1).time(end)/dt])
% title('IC spike')
% xlim([2500 3200])

figure;
for trialToPlot = 1:20
    plot(temp(2).time,temp(trialToPlot).R_V(:,1) + 20*(trialToPlot-1),'color', 	[0, 0.4470, 0.7410]); hold on;
    plot(temp(2).time,temp(trialToPlot).Exc_V(:,1) + 20*(trialToPlot-1),'color', [0.8500, 0.3250, 0.0980]);
end
legend('R','E')
xlabel('time')
ylabel('V')
ylim([-75 350])
yticks([-50:20:350])
yticklabels([1:20])