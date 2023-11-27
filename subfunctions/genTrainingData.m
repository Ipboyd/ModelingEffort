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
% ICdir = 'ICSimStim\bird\full_grids\BW_0.004 BTM_3.8 t0_0.1 phase0.4900\s7_STRFgain1.00_20210107-173527';
ICdir = 'ICSimStim\mouse\full_grids\BW_0.009 BTM_3.8 t0_0.1 phase0.499\s1.5_STRFgain0.50_20200514-181040';

addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = 'training 001 mouseTuning';

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
options.STRFgain = extractBetween(ICdir,'gain','_2020');
options.plotRasters = 0;
options.time_end = size(spks,3)*dt; %ms;
options.locNum = [];
options.parfor_flag = 1;
[snn_out,s] = birdNetwork(study_dir,varies,netcons,options);

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
    [data(z).perf,data(z).fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    
%     [dataOld(z).perf,dataOld(z).fr] = postProcessData(snn_out,trialStart,trialEnd,options);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results
% figure;
% vizNetwork(s,0,'C','Exc')

% data = dataOld;
% plotPerformanceGrids;
% 
if length(subz) == 24
    options.subPops = {'C'};
    plotPerformanceGrids_new;
    
    subplot(1,3,2); imagesc(netcons.xrNetcon);
    colorbar; caxis([0 1])
    
    subplot(1,3,3); imagesc(netcons.rcNetcon);
    colorbar; caxis([0 1])
end
%% Smooth and delay spike trains
t = (0:dt:500)/1000; % 0-500ms
tau = 0.005; % second
kernel = t.*exp(-t/tau);

% amount of delay between input and output, in units of taps
% 1,9,17 for bird data
NumDelayTapsL0 = 1; %E
NumDelayTapsL1 = 9; %R,X
NumDelayTapsL2 = 17; %C

snn_spks = [];
% snn_spks.IC.delay = NumDelayTapsL2; %not strictly necessary for now 
snn_spks.IC.delay = 0;
snn_spks.E.delay = NumDelayTapsL0;
snn_spks.R.delay = NumDelayTapsL1;
snn_spks.X.delay = NumDelayTapsL1;
snn_spks.C.delay = NumDelayTapsL2;

for variation = 1:numVaried
    disp(variation);
    % for each varied parameter
    
    % output data
    Cspks = [snn_out(variation:numVaried:end).C_V_spikes];

    % intermediate neurons
    for i = variation:numVaried:20*numVaried %number of trials
        snn_spks(variation).R.raw(i,:,:) = snn_out(i).R_V_spikes;
        snn_spks(variation).X.raw(i,:,:) = snn_out(i).X_V_spikes;
        snn_spks(variation).E.raw(i,:,:) = snn_out(i).Exc_V_spikes;
    end

    % combine across trials, delay, smooth with kernel, remove zero-padded length
    n = padToTime/dt * length(subz);
    for songn = 1:2
        delay = snn_spks(1).IC.delay;
        m = n-delay;
        snn_spks(variation).IC.psth.song{songn} = squeeze(sum(spks((1:10) + 10*(songn-1),:,:)));
        snn_spks(variation).IC.psth.song{songn} = snn_spks(variation).IC.psth.song{songn}(:,1:end-delay);
        delayedSpks = snn_spks(variation).IC.psth.song{songn};
        snn_spks(variation).IC.smoothed.song{songn} = conv2(delayedSpks',kernel');
        snn_spks(variation).IC.smoothed.song{songn} =  snn_spks(variation).IC.smoothed.song{songn}(1:m,:);

        for neuron = {'R','X','E'}
            delay = snn_spks(1).(neuron{1}).delay;
            m = n-delay;
            snn_spks(variation).(neuron{1}).psth.song{songn} = squeeze(sum(snn_spks(variation).(neuron{1}).raw((1:10) + 10*(songn-1),:,:)));
            snn_spks(variation).(neuron{1}).psth.song{songn} = snn_spks(variation).(neuron{1}).psth.song{songn}(1+delay:end,:);
            delayedSpks = snn_spks(variation).(neuron{1}).psth.song{songn};
            snn_spks(variation).(neuron{1}).smoothed.song{songn} = conv2(delayedSpks,kernel');
            snn_spks(variation).(neuron{1}).smoothed.song{songn} =  snn_spks(variation).(neuron{1}).smoothed.song{songn}(1:m,:);
        end

        delay = snn_spks(1).C.delay;
        m = n-delay;
        snn_spks(variation).C.psth.song{songn} = sum(Cspks(:,(1:10) + 10*(songn-1)),2);
        snn_spks(variation).C.psth.song{songn} = snn_spks(variation).C.psth.song{songn}(1+delay:end,:);
        delayedSpks = snn_spks(variation).C.psth.song{songn};
        snn_spks(variation).C.smoothed.song{songn} = conv(delayedSpks',kernel);
        snn_spks(variation).C.smoothed.song{songn} = snn_spks(variation).C.smoothed.song{songn}(1:m);
    end
end
%% visualizations
non0chans = find(sum(spks(1,:,:),3));
for variation = 1:numVaried
    % for each varied parameter
    if length(non0chans)==1
        % all neurons, one channel
        channel = non0chans;
        for songn = 1:2
            figure;
            for neuron = {'IC','E','R','X'}
                current_psth = snn_spks(variation).(neuron{1}).smoothed.song{songn}(:,channel);
                plot(current_psth,'linewidth',1.5,'linestyle','-'); hold on;
            end
            plot(snn_spks(variation).C.smoothed.song{songn},'linewidth',1.5,'linestyle','-.');
            xlim([2500 3200])
            legend('in','E','R','X','out')
        end
    elseif length(non0chans) == 2
        % input & output only, 2 channels
        chan1 = non0chans(1);
        chan2 = non0chans(2);
        for songn = 1:2
            figure;
            chan1_psth = snn_spks(variation).IC.smoothed.song{songn}(:,chan1);
            plot(chan1_psth,'linewidth',1.5,'linestyle','-'); hold on;
            chan2_psth = snn_spks(variation).IC.smoothed.song{songn}(:,chan2);
            plot(chan2_psth,'linewidth',1.5,'linestyle','-'); hold on;
            summed_psth = chan1_psth+chan2_psth;
            plot(summed_psth,'linewidth',1.5,'linestyle','-'); hold on;
            plot(snn_spks(variation).C.smoothed.song{songn},'linewidth',1.5,'linestyle','-.');
            xlim([2500 3200])
            legend('IC 1st channel','IC 2nd channel','chan1+chan2','out')
        end
    else
        for songn = 1:2
%             psth = snn_spks(variation).IC.smoothed.song{songn};
            psth = snn_spks(variation).R.smoothed.song{songn};
            psth_converge = sum(psth.*netcons.rcNetcon',2);
            psth_out = snn_spks(variation).C.smoothed.song{songn};

            MAE_RC(songn) = mean(abs(psth_converge(1:length(psth_out))-(psth_out'))./(psth_out'+1E-6)); %mean absolute % error
%             xrInhSpot(xrInhSpotIdx).MAE_RC = MAE_RC;
            
            figure;
            plot(psth,'linewidth',1.5); hold on;
            plot(psth_converge,'linewidth',1.5);
            plot(psth_out,'linewidth',1.5,'linestyle','-.');
            xlim([2500 3200])
            legend('IC 1st channel','IC 2nd channel','IC 3nd channel','IC 4nd channel','sum','out')
            title(['R->C Mean absolute % error = ' num2str(MAE_RC(songn)*100) '%'])


            
            % compare PSTH_IC to PSTH_R
            % R =? R_est = Relu(X-WX)
            R = snn_spks(variation).R.smoothed.song{songn};
            R_est = (snn_spks(variation).IC.smoothed.song{songn} * (-1*netcons.xrNetcon + eye(4)) );
            R_est = max(R_est,0);
            Rlen = size(R_est,1);
            
%             xrInhSpot(xrInhSpotIdx).MAE_R(:,songn) = 100 * mean( abs((R_est - R(1:Rlen,:))) ./ (R(1:Rlen,:)+1E-6) );

            figure;
            cmap = [[0, 0.4470, 0.7410];[0.8500, 0.3250, 0.0980];[0.9290, 0.6940, 0.1250];[0.4940, 0.1840, 0.5560]];
            RLen = min(length(R),length(R_est));
            for i = 1:4
                plot(R(:,i),'color',cmap(i,:)); hold on;
                plot(R_est(:,i),'--','color',cmap(i,:))
                MAE_XR(i) = mean(abs(R(1:RLen,i) - R_est(1:RLen,i)) ./ (R(1:RLen,i) + 1E-6)) * 100;
            end
            annotation('textbox',[.2 .6 .3 .3],'string', sprintf('MAE_XR, channel %i: %.02f \n',[(1:4); MAE_XR]),'FitBoxToText','on')
            title('PSTH, R vs R_{est}')
            legend('chan1','est','2','est','3','est','4','est')
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%% save results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if save_flag
    input_training = [snn_spks.IC.smoothed.song{1}; snn_spks.IC.smoothed.song{2}];
    output_training = [snn_spks.C.smoothed.song{1}, snn_spks.C.smoothed.song{2}]';
    perf_data = data;
    trialLen = padToTime - NumDelayTapsL2;
    name = sprintf('training_set_%i_noU',trainingSetNum);
    save(['SNN_optimization' filesep name '.mat'],'input_training','output_training',...
        'perf_data','netcons','network_params','options','trialLen','s','ICdir','varies');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end

%%
% Rspks = snn_spks(variation).R.raw;
% Xspks = snn_spks(variation).X.raw;
% Espks = snn_spks(variation).E.raw;
% 
% figure;
% plotSpikeRasterFs(logical(Xspks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('X spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(Rspks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('R spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(squeeze(spks(:,1,:))),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('IC spike')
% xlim([2500 3200])

% figure;
% scaleFactor = 30;
% for trialToPlot = 1:20
%     plot(snn_out(2).time,snn_out(trialToPlot).R_V(:,1) + scaleFactor*(trialToPlot-1),'color', [0, 0.4470, 0.7410]); hold on;
%     plot(snn_out(2).time,snn_out(trialToPlot).X_V(:,2) + scaleFactor*(trialToPlot-1),'color', [0.8500, 0.3250, 0.0980]);
% end
% legend('R','X')
% xlabel('time')
% ylabel('V')
% ylim([-80 scaleFactor*20-70])
% yticks([-50:scaleFactor:scaleFactor*20-50])
% yticklabels([1:scaleFactor])