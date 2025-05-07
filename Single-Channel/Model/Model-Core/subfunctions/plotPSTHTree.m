function plotPSTHTree(data,configName,options)
% calculate performance and FR for *a single spot on the spatial grid*
% input:
%   data structure with length = #sims, containing voltage information of
%   [time x spatial channel]. Time dimension contains information from each
%   spot in the spatial grid. Extract this information with
%   options.trialStart and options.trialEnd.
% output:
%   data
%       spatial channels,trials,time
% no plotting function for now

trial_length = options.trial_length;
dt = options.dt;

jump = length(data.spks.On);

% network properties
%popNames = fieldnames(data.fr);
% popNames = {'X','ROn','On'};
popNames = {'S1OnOff','R1On','R2On'};

nPops = numel(popNames); 

nChans = length(fieldnames(data.perf.On));
annotConfig = configName(end-3:end);

% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end

[subplot_locs,n_rows,n_cols] = detSubplotLocs(popNames);

%figure('unit','inches','position',[6 3 6 5]);

subSpks = data.spks;
subPC = data.perf;
subFR = data.fr;

for vv = 1:jump % for each varied parameter
  
    for ch = 1:nChans

        figure('unit','inches','position',[6 3 6 5]);

        for currentPop = 1:nPops

            ch_num = ch;

            plotSubPSTH(subSpks.(popNames{currentPop})(vv).(['channel' num2str(ch_num)]),...
                subPC.(popNames{currentPop}).(['channel' num2str(ch_num)])(vv),...
                subFR.(popNames{currentPop}).(['channel' num2str(ch_num)])(vv),...
                trial_length,dt,popNames{currentPop},subplot_locs(currentPop),[n_rows n_cols]);
        end

        % make ylim the same across all subplots
        psths = findobj(gcf,'type','line');
        if vv > 1
            ymax = max([psths.YData]);
            plts = findobj(gcf,'type','axes');
            for p = 1:length(plts)
                plts(p).YLim = [0 ymax];
            end
        end
        figName = sprintf('%s_CH%i_set%s_PSTH',configName,ch,num2str(vv));
        
        annotation('textbox',[.6 .82 .1 .1], ...
            'String',[annotConfig, ', CH' num2str(ch) ],'EdgeColor','none','FontSize',20)
        
        saveas(gcf,[figName '.png']);
        % savefig(gcf,[figName '.fig']);
        %clf;
    end

end

end

function plotSubPSTH(raster,pc,fr,trial_length,dt,unit,subplot_loc,subplot_dims)

PCstr = ['PC = ' num2str(round(pc)) '%'];

% ind2sub counts down per column first,`
[c,r] = ind2sub(subplot_dims + [1 0],subplot_loc);
r = subplot_dims(1) + 1 - r;
ypos = 0.06 + 0.215*(r-1);
y = 0.12;

xpos = 0.06 + 0.23*(c-1);
x = 0.22;

subplot('position',[xpos ypos x y]); hold on;

% plot target responses on top of each other

t_bin = 20; % in [ms] 
t_vec = (0:t_bin:trial_length)/dt;

colors = {'k','r'};
for tid = 1:2
    [~,spk_inds] = find(raster((1:10) + 10*(tid-1),:));
    PSTH = histcounts(spk_inds,t_vec);
    plot(dt*t_vec(1:end-1),PSTH,colors{tid});
end
xlim([0 t_vec(end)*dt]);
title({unit,[PCstr,[', FR = ' num2str(fr)]]},'fontweight','normal','fontsize',8); set(gca,'xtick',[],'ytick',[])

end
