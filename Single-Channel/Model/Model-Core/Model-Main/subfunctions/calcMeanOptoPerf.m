function calcMeanOptoPerf(results,nVaried,simDataDir)

[pc,fr]= plotParamvsPerf_1D(results,nVaried);

% performance
pc_trials = struct2cell(pc);

ctrl_mean = cellfun(@(x) mean(x(:,1)),pc_trials);
laser_mean = cellfun(@(x) mean(x(:,end)),pc_trials);

ctrl_se = cellfun(@(x) std(x(:,1)), pc_trials);%/sqrt(numel(x(:,1))),pc_trials);
laser_se = cellfun(@(x) std(x(:,end)), pc_trials);%/sqrt(numel(x(:,end))),pc_trials);

figure('unit','inches','position',[5 5 3 3]);
bar((1:4)-.2,ctrl_mean,0.4,'facecolor','none','linewidth',2); hold on;
bar((1:4)+.2,laser_mean,0.4,'facecolor','k','linewidth',2);
xlim([0.4 4.6]);

errorbar((1:4)-.2,ctrl_mean,ctrl_se,'color','k','linestyle','none','linewidth',1); hold on;
errorbar((1:4)+.2,laser_mean,laser_se,'color','k','linestyle','none','linewidth',1);

p_vals = cellfun(@(x) ranksum(x(:,1),x(:,2)),pc_trials)
groups = mat2cell([[1:4]'-0.2 [1:4]'+0.2],[ 1 1 1 1 ]);
ylim([50 100]);

addpath('sigstar-master\');

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