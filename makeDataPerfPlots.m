% bar plot for different hotspot types - Used in Figure 2 of modelling
% paper

load('singleunits_diffperfs.mat')
load('singleunits.mat')

aa = singleunits_comp(strcmp({singleunits_comp.stim},'Clean'));

bb = singleunits(strcmp({singleunits.stim},'Clean'));

perfs = [[aa.ctrl_perf_SPIKE];[aa.ctrl_perf_ISI];[aa.ctrl_perf_RISPIKE]];

% sort hotspots by the order of their performance measures
[~,ind] = sort(perfs,1,'descend');
[types,~,type_ind] = unique(ind','rows');

% types 1 and 3 are rate based, types 2 and 4 are timing-based
% lump 4 in with 2
type_ind(type_ind == 4) = 2;

% how is each group affected by opto?

% aa = singleunits_comp(strcmp({singleunits_comp.stim},'Masked'));
perf_ctrl = [[aa.ctrl_perf_SPIKE];[aa.ctrl_perf_ISI];[aa.ctrl_perf_RISPIKE]];
perf_laser = [[aa.laser_perf_SPIKE];[aa.laser_perf_ISI];[aa.laser_perf_RISPIKE]];

% [~,ind] = max(perf_ctrl,[],1);

names = {'Type 1','Type 2','Type 3'};

for i = 1:3
    figure('unit','inches','position',[3 3 2.2 2.5]);
    temp_ctrl = perf_ctrl(:,type_ind==i);
    temp_laser = perf_laser(:,type_ind==i);



    bar((1:3)-0.2,mean(temp_ctrl,2),0.4,'facecolor','none','linewidth',2); hold on;
    bar((1:3)+0.2,mean(temp_laser,2),0.4,'facecolor','k','linewidth',2); hold on;
    errorbar((1:3)-0.2,mean(temp_ctrl,2),std(temp_ctrl,[],2)/sqrt(size(temp_ctrl,2)),'k','linestyle','none','linewidth',2);
    errorbar((1:3)+0.2,mean(temp_laser,2),std(temp_laser,[],2)/sqrt(size(temp_laser,2)),'k','linestyle','none','linewidth',2);

    title([names{i} ' clean hotspots'])
    legend('Control','Laser')
    set(gca,'xtick',1:3,'xticklabel',{'SPIKE','ISI','RI-SPIKE'},'fontsize',8);
    ylim([50 100]); xlim([0.5 3.5]); ylabel('Performance');
    ytickformat('percentage');
    savefig(gcf,[names{i} '-distance control vs laser']);
end

% FR for different spots
figure('unit','inches','position',[3 3 2.2 2.5]);

for i = 1:3
    temp_ctrl = [bb(type_ind==i).ctrl_driven_FR];
    temp_laser = [bb(type_ind==i).laser_driven_FR];

    bar(i-0.2,mean(temp_ctrl,2),0.4,'facecolor','none','linewidth',2); hold on;
    bar(i+0.2,mean(temp_laser,2),0.4,'facecolor','k','linewidth',2); hold on;
    errorbar(i-0.2,mean(temp_ctrl,2),std(temp_ctrl,[],2)/sqrt(size(temp_ctrl,2)),'k','linestyle','none','linewidth',2);
    errorbar(i+0.2,mean(temp_laser,2),std(temp_laser,[],2)/sqrt(size(temp_laser,2)),'k','linestyle','none','linewidth',2);
     
end

title(['Type hotspots FR']);
set(gca,'xtick',1:3,'xticklabel',{'SPIKE-best','Timing-based','Rate-based'},'fontsize',8); xlabel('Type');
xlim([0.5 3.5]); ylabel('FR (Hz)');
savefig(gcf,[names{i} '-distance hotspots FR']);

% histogram of timing and firing rate-based hotspots

ratios = perf_ctrl(3,:) ./ perf_ctrl(2,:);

figure('unit','inches','position',[4 4 2 1.6]); histogram(ratios,0.5:0.05:1.5,'facecolor','k');
xlabel('Timing-based to rate-based performance ratio'); set(gca,'fontsize',8);
savefig(gcf,'timing vs rate histogram')

% FR vs ratio correlation
FR_ctrl = [bb.ctrl_driven_FR];
FR_laser = [bb.laser_driven_FR];

figure('unit','inches','position',[4 4 2.2 1.9]);
scatter(ratios,FR_ctrl,'filled','k');
hold on;
ratios_laser = perf_laser(3,:) ./ perf_laser(2,:);
scatter(ratios_laser,FR_laser,'filled','r');
legend('Control','Laser');
xlabel('Timing-based to rate-based performance ratio'); ylabel('FR (Hz)');
savefig(gcf,'FR vs ratio')

[R_ctrl,P_ctrl] = corrcoef(ratios,FR_ctrl)
[R_laser,P_laser] = corrcoef(ratios_laser,FR_laser)


