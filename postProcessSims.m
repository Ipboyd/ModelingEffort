numVaried = length(snn_out)/(20*nSims);

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
    [pc,fr]= plotParamvsPerf_1D(results,numVaried);

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

[pc,fr] = plotParamvsPerf_1D(results,numVaried)
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


if numVaried >= 10
    plotPerfvsParams('R2On',data,varies,simDataDir)
    close all;
end

% trial similarity and RMS difference
clearvars spks TS RMS

for nS = 1:nSims
    for nV = 1:numVaried

        spks = {snn_out((nV + (nS-1)*numVaried) : numVaried*nSims : end).R2On_V_spikes};
        for n = 1:20
            spks{n} = find(spks{n})/10000 - 0.3;
        end
        spks = reshape(spks,10,2);
        [TS(nV,nS),RMS(nV,nS)] = calcTrialSim(spks);

    end
end

save([simDataDir filesep 'TS_RMS_R2On.mat'],'TS','RMS');