nVaried = length(snn_out)/(20*nSims);

% load ICfiles struct just for the names of the configs
load('ICfiles.mat'); subz = 1:24;

% calculate performance
data = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
configName = cellfun(@(x) strsplit(x,'_'),{ICfiles(subz).name}','UniformOutput',false);
configName = vertcat(configName{:}); configName = configName(:,1);
options.variedField = strrep(expVar,'-','_');

annotTable = createSimNotes(snn_out,simDataDir,options);

% save C spikes and varied params to struct
names = snn_out(1).varied; results = struct;
for i = 1:length(snn_out)
    results(i).R2On_V_spikes = snn_out(i).R2On_V_spikes;
    for t = 1:length(names)
        results(i).(names{t}) = snn_out(i).(names{t});
    end
end
results(1).model = snn_out(1).model; save([simDataDir filesep 'R2On_results.mat'],'results');

tic;
if ~isempty(options.locNum)
    trialStart = 1; trialEnd = padToTime/dt;
    figName = [simDataDir filesep configName{options.locNum}(1:end-4)];
    [data.perf,data.fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options);
    plotPSTHTree(snn_out,s,trialStart,trialEnd,figName,options)
else
    for z = 1:24
        trialStart = PPtrialStartTimes(z);
        trialEnd = PPtrialEndTimes(z);
        figName = [simDataDir filesep configName{z}];
        [data(z).perf,data(z).fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
        plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options);
    end
end
toc;
% close all;

if nSims == 5
    [pc,fr]= plotParamvsPerf_1D(results,nVaried);

    % performance
    pc_trials = struct2cell(pc);

    ctrl_mean = cellfun(@(x) mean(x(:,1)),pc_trials);
    laser_mean = cellfun(@(x) mean(x(:,end)),pc_trials);

    ctrl_se = cellfun(@(x) std(x(:,1))/sqrt(numel(x(:,1))),pc_trials);
    laser_se = cellfun(@(x) std(x(:,end))/sqrt(numel(x(:,end))),pc_trials);

    figure('unit','inches','position',[5 5 3 3]);
    bar((1:4)-.2,ctrl_mean,0.4,'facecolor','none','linewidth',2); hold on;
    bar((1:4)+.2,laser_mean,0.4,'facecolor','k','linewidth',2);
    xlim([0.4 4.6]);

    errorbar((1:4)-.2,ctrl_mean,ctrl_se,'color','k','linestyle','none','linewidth',1); hold on;
    errorbar((1:4)+.2,laser_mean,laser_se,'color','k','linestyle','none','linewidth',1);

    p_vals = cellfun(@(x) ranksum(x(:,1),x(:,2)),pc_trials)
    groups = mat2cell([[1:4]'-0.2 [1:4]'+0.2],[ 1 1 1 1 ]);
    ylim([50 100]);

    sigstar(groups,p_vals);

    set(gca,'xticklabels',{'SPIKE','ISI','RI-SPIKE','Spike count'},'xtick',1:4,'fontsize',8);
    ytickformat('percentage');
    ylabel('Performance'); legend('Control','Laser');
    saveas(gcf,[simDataDir filesep 'opto_performance_results.fig']);

    % firing rate
    ctrl_mean = mean(fr(:,1));
    laser_mean = mean(fr(:,end));

    ctrl_se = std(fr(:,1))/sqrt(5);
    laser_se = std(fr(:,end))/sqrt(5);

    figure('unit','inches','position',[5 5 2 3]);
    bar(1-.2,ctrl_mean,0.4,'facecolor','none','linewidth',2); hold on;
    bar(1+.2,laser_mean,0.4,'facecolor','k','linewidth',2);
    xlim([0.4 1.6]);

    errorbar(1-.2,ctrl_mean,ctrl_se,'color','k','linestyle','none','linewidth',1); hold on;
    errorbar(1+.2,laser_mean,laser_se,'color','k','linestyle','none','linewidth',1);

    ylim([0 60]);
    set(gca,'xticklabels',{'Control','Laser'},'xtick',[],'fontsize',8);
    ylabel('Firing rate (Hz)'); legend('Control','Laser');
    saveas(gcf,[simDataDir filesep 'opto_FR_results.fig']);
end

[pc,fr] = plotParamvsPerf_1D(results,nVaried)
save([simDataDir filesep 'perf_fr_R2On.mat'],'pc','fr')

% x = varies(varied_param).range;
% % x = x(2,:); x(1) = 0;
%
% figure;
% plot(x,pc.SPIKE); hold on; plot(x,pc.ISI); plot(x,pc.RISPIKE)
% legend('SPIKE','ISI','RI-SPIKE'); ylim([50 100]);
% xlabel([varies(varied_param).conxn '_{' varies(varied_param).param '}']);
% title(['Performance vs ' varies(varied_param).conxn '_{' varies(varied_param).param '}']);
% ytickformat('percentage'); ylabel('Performance')
%
% savefig(gcf,[simDataDir filesep 'perfs_vs_params.fig'])
% saveas(gcf,[simDataDir filesep 'perfs_vs_params.png'])


if nVaried >= 10
    plotPerfvsParams('R2On',data,varies,simDataDir)
    close all;
end

% trial similarity and RMS difference
clearvars spks TS RMS

for nS = 1:nSims
    for nV = 1:nVaried

        spks = {snn_out((nV + (nS-1)*nVaried) : nVaried*nSims : end).R2On_V_spikes};
        for n = 1:20
            spks{n} = find(spks{n})/10000 - 0.3;
        end
        spks = reshape(spks,10,2);
        [TS(nV,nS),RMS(nV,nS)] = calcTrialSim(spks);

    end
end

save([simDataDir filesep 'TS_RMS_R2On.mat'],'TS','RMS');

%% convert peakDrv to samples (10000 Hz)
close all;
load('peakDrv_target.mat');
figure(1);

% clearvars envelope
% fs = 44100; t = 1.5;
% envelope(:,1) = [ abs(sin(2*pi*AM_freqs(a)*(0:round(fs*1.5))'/fs)) ];
% envelope(:,2) = [ abs(sin(2*pi*AM_freqs(a)*(0:round(fs*1.5))'/fs)) ];
% stim_vec = (0:(size(envelope,1)-1))/fs;

t_bin = 20; %ms
t_vec = 0:t_bin:(3500-1);

pops = {'On','R1On','R2On'};
clearvars spks
for p = 1:length(pops)
    spks = sprintf('%s_V_spikes',pops{p});
    for a = 1:2
        figure(1);
        subplot(2,1,a);

        % convert peakDrv to ms and add 250ms (to account for zeropadding in Am
        % stimuli)
        peakDrv_ms = [round(peakDrv_target{a}(2:end),1)]+250;
        temp = [];
        % get spiketime indexes for each trial
        for t = (1:10)+(a-1)*10
            temp = [temp; find(snn_out(t).(spks))];
        end

        % convert to PSTH
        PSTH = histcounts(temp,0:t_bin*10:35000);
        plot(t_vec,PSTH,'linewidth',1); hold on;
        if p == length(pops)
            plot([peakDrv_ms;peakDrv_ms],[0 20]'.*ones(size(peakDrv_ms)),'--r','linewidth',1);
        end
        %     plot(stim_vec*1000,20*envelope(:,a),'b');
        ylabel(sprintf('Target %i Hz',a))

        % get activity within 150ms of each peakDrv event
        peakAct = [];
        for d = 1:length(peakDrv_ms)
            temp2 = temp/10 - peakDrv_ms(d);
            peak_vec = -150 : 6 : 150;
            peakAct(d,:) = histcounts(temp2,peak_vec);
        end
        figure(2);
        subplot(2,1,a);
        plot(peak_vec(1:end-1),mean(peakAct),'linewidth',1); hold on;
        if p ==  length(pops)
            plot([0 0],[0 10],'r');
        end
        ylabel(sprintf('Target %i Hz',a))
        %ylim([3 7])
    end
end

figure(1);
xlabel('Time (ms)');
legend(pops);
savefig(gcf,[simDataDir filesep 'AM response.fig']);

figure(2);
xlabel('Time from peakDrv (ms)');
legend(pops);
savefig(gcf,[simDataDir filesep 'peakDrv.fig']);

% %% Spike-triggered averages
% close all;
% 
% pops = {'R2On'};%{'On','R1On','R2On'};
% clearvars spks
% 
% fs_stim = 200000/5;
% y1_ds = downsample(y1,5)'; y2_ds = downsample(y2,5)';
% tlen_STA = .10; % [seconds]
% 
% figure;
% for p = 1:length(pops)
%     spks = sprintf('%s_V_spikes',pops{p});
%     for a = 1:2 % for each target
%         subplot(2,1,a);
%         temp = [];
%         % get spiketimes for each trial
%         for t = (1:10)+(a-1)*10
%             temp = [temp; find(snn_out(t).(spks))/10000];
%         end
% 
%         % for each spike time, get last 150 ms of target stimulus
% 
%         % convert to indexes from target
%         inds = round(temp * fs_stim); inds(inds < round(fs_stim*tlen_STA) | inds > length(y1_ds)) = [];
% 
%         t_STA = (-round(fs_stim*tlen_STA):0)/fs_stim;
%         inds = (-round(fs_stim*tlen_STA):0) + inds;
%         STA = [];
%         for i = 1:size(inds,1)
%             STA = cat(1,STA,eval(['y' num2str(a) '_ds(inds(i,:))']));
%         end
%         STA_mean = mean(STA);
% 
%         plot(t_STA,STA_mean,'linewidth',1); hold on;
%     end
% end
% sgtitle('Direct STA');
% 
% savefig(gcf,[simDataDir filesep 'direct STA.fig']);
% close;
% 
% %% STA from envelope
% 
% load('target_envelopes.mat','y_env');
% fs_stim = 200000/5;
% tlen_STA = .150; % [seconds]
% y1_env_ds = downsample(y_env{1}',5); y1_env_ds(y1_env_ds<0)=0;
% y2_env_ds = downsample(y_env{2}',5); y2_env_ds(y2_env_ds<0)=0;
% 
% y1_dB = 20*log10(y1_env_ds);
% y2_dB = 20*log10(y2_env_ds);
% 
% 
% pops = {'R2On'};%{'On','R1On','R2On'};
% clearvars spks
% 
% figure;
% for p = 1:length(pops)
%     spks = sprintf('%s_V_spikes',pops{p});
%     for a = 1:2 % for each target
%         subplot(2,1,a);
%         temp = [];
%         % get spiketimes for each trial
%         for t = (1:10)+(a-1)*10
%             temp = [temp; find(snn_out(t).(spks))/10000];
%         end
% 
%         % for each spike time, get last 150 ms of target stimulus
% 
%         % convert to indexes from target
%         inds = round(temp * fs_stim); inds(inds < round(fs_stim*tlen_STA) | inds > length(y1_env_ds)) = [];
% 
%         t_STA = (-round(fs_stim*tlen_STA):0)/fs_stim;
%         inds = (-round(fs_stim*tlen_STA):0) + inds;
%         STA = []; STA_dB=[];
%         for i = 1:size(inds,1)
%             STA = cat(1,STA,eval(['y' num2str(a) '_env_ds(inds(i,:))']));
%             STA_dB = cat(1,STA_dB,eval(['y' num2str(a) '_dB(inds(i,:))']));
%         end
%         STA_mean = mean(STA);
%         STA_dB(isinf(STA_dB)) = nan;
%         STA_dB_mean = mean(STA_dB,'omitnan');
% 
%         yyaxis left;
%         plot(t_STA,STA_mean,'linewidth',1);
%         yyaxis right;
%         plot(t_STA,STA_dB_mean,'linewidth',1);
%     end
% end
% sgtitle('Envelope STA');
% % legend('Linear','dB');
% % legend(pops)
% 
% savefig(gcf,[simDataDir filesep 'envelope STA.fig']);

