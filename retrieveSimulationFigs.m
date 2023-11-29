cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid')
dynasimPath = '../DynaSim';
addpath('mechs'); addpath('fixed-stimuli'); addpath(genpath('ICSimStim'));
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');

[folder] = uigetdir('simData');

figs = dir(fullfile(folder,'*.fig'));
figs(1:2) = [];

FR_pops = struct;
pc_pops = struct;

nctrl = 1; nopto = 1;
for n = 1:10

    simnum = str2double(extractBetween(figs(n).name,'set','.fig'));
    openfig(fullfile(folder,figs(n).name));

    [pc,fr] = getPerffromRaster;

    units = fieldnames(fr);
    for u = 1:length(units)
        if rem(simnum,2) == 1 % odd = control sim
            FR_pops.(units{u})(1,nctrl) = fr.(units{u});
            pc_pops.(units{u})(1,nctrl) = pc.(units{u}).SPIKE;
        else
            FR_pops.(units{u})(2,nopto) = fr.(units{u});
            pc_pops.(units{u})(2,nopto) = pc.(units{u}).SPIKE;
        end
    end

    if rem(simnum,2) == 1 % odd = control sim
        nctrl = nctrl + 1;
    else
        nopto = nopto + 1;
    end
    close;
end


figure('unit','inches','position',[3 3 3 1.4]);
subplot(1,2,1);
bar(1:3,[mean(pc_pops.On(1,:)),mean(pc_pops.R1On(1,:)),mean(pc_pops.R2On(1,:))],'facecolor','none','linewidth',1); hold on;
errorbar([mean(pc_pops.On(1,:)),mean(pc_pops.R1On(1,:)),mean(pc_pops.R2On(1,:))],...
    [std(pc_pops.On(1,:))/5,std(pc_pops.R1On(1,:))/5,std(pc_pops.R2On(1,:))/5],'k','linestyle','none')
ylim([50 100]); ytickformat('percentage'); xlim([0.5 3.5]); set(gca,'xticklabel',{'IC','Int','Out'},'fontsize',8); ylabel('Performance');

subplot(1,2,2);
bar(1:3,[mean(FR_pops.On(1,:)),mean(FR_pops.R1On(1,:)),mean(FR_pops.R2On(1,:))],'facecolor','none','linewidth',1); hold on;
errorbar([mean(FR_pops.On(1,:)),mean(FR_pops.R1On(1,:)),mean(FR_pops.R2On(1,:))],...
    [std(FR_pops.On(1,:))/5,std(FR_pops.R1On(1,:))/5,std(FR_pops.R2On(1,:))/5],'k','linestyle','none')
ylim([0 80]); xlim([0.5 3.5]); set(gca,'xticklabel',{'IC','Int','Out'},'fontsize',8); ylabel('FR (Hz)')
savefig(gcf,'Timing-based perf and FR vs layer');
saveas(gcf,'Timing-based perf and FR vs layer.png');

% compare firing rates from PVs control vs laser

figure('unit','inches','position',[3 3 2.5 1.5]);
subplot(1,2,1);
bar(1:2,[mean(FR_pops.S1On(1,:)),mean(FR_pops.S1On(2,:))],'facecolor','none','linewidth',1); hold on;
errorbar([mean(FR_pops.S1On(1,:)),mean(FR_pops.S1On(2,:))],...
    [std(FR_pops.S1On(1,:))/5,std(FR_pops.S1On(2,:))/5],'k','linestyle','none')
ylim([0 50]); xlim([0.5 2.5]); set(gca,'xticklabel',{'Control','Opto'},'fontsize',8); ylabel('FR (Hz)')
title('Int PV FR','fontweight','normal','fontsize',10)

subplot(1,2,2);

bar(1:2,[mean(FR_pops.S2On(1,:)),mean(FR_pops.S2On(2,:))],'facecolor','none','linewidth',1); hold on;
errorbar([mean(FR_pops.S2On(1,:)),mean(FR_pops.S2On(2,:))],...
    [std(FR_pops.S2On(1,:))/5,std(FR_pops.S2On(2,:))/5],'k','linestyle','none')
ylim([0 50]); xlim([0.5 2.5]); set(gca,'xticklabel',{'Control','Opto'},'fontsize',8); ylabel('FR (Hz)')
title('Output PV FR','fontweight','normal','fontsize',10)
savefig(gcf,'Timing-based PV FR control vs layer by layer');
saveas(gcf,'Timing-based PV FR control vs layer by layer.png');
