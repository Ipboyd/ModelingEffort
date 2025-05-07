function plotRasterTree(data,configName,options)
% calculate performance and FR for *a single spot on the spatial grid*
% input:
%   data structure with length = #sims, containing voltage information of
%   [time x spatial channel]. Time dimension contains information from each
%   spot in the spatial grid. Extract this information with
%   options.trialStart and options.trialEnd.

trial_length = options.trial_length;
dt = options.dt;

jump = length(data.spks.On);

% network properties
%popNames = fieldnames(data.fr);
% popNames = {'SOnOff','ROn'};
popNames = {'S1OnOff','R1On','R2On'};



nPops = numel(popNames); 

nChans = length(fieldnames(data.perf.On));

annotConfig = configName(end-3:end);

% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end

[subplot_locs,n_rows,n_cols] = detSubplotLocs(popNames);

% locs = {'90','45','0','-90'};
%figure('unit','inches','position',[6 3 6 5]);

subSpks = data.spks;
subPC = data.perf;
subFR = data.fr;

for vv = 1:jump % for each varied parameter
    
    for ch = 1:nChans

        figure('unit','inches','position',[6 3 6 5]);

        for currentPop = 1:nPops

            ch_num = ch;

            plotSubRaster(subSpks.(popNames{currentPop})(vv).(['channel' num2str(ch_num)]),...
                subPC.(popNames{currentPop}).(['channel' num2str(ch_num)])(vv),...
                subFR.(popNames{currentPop}).(['channel' num2str(ch_num)])(vv),...
                trial_length,dt,popNames{currentPop},subplot_locs(currentPop),[n_rows n_cols]);
        end

        figName = sprintf('%s_CH%i_set%s',configName,ch,num2str(vv));

        annotation('textbox',[.6 .82 .1 .1], ...
            'String',[annotConfig, ', CH' num2str(ch)],'EdgeColor','none','FontSize',20)

        saveas(gcf,[figName '.png']);
        % savefig(gcf,[figName '.fig']);
        %clf
    end

end

end

function plotSubRaster(raster,pc,fr,trial_length,dt,unit,subplot_loc,subplot_dims)

% ind2sub counts down per column first,`
[c,r] = ind2sub(subplot_dims + [1 0],subplot_loc);
r = subplot_dims(1) + 1 - r;
ypos = 0.06 + 0.215*(r-1);
y = 0.12;

xpos = 0.06 + 0.23*(c-1);
x = 0.22;

PCstr = ['PC = ' num2str(round(pc)) '%'];

subplot('position',[xpos ypos x y]);

% plot both targets
plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
xlim([0 trial_length/dt]); ylim([0.5 20.5]);
title({unit,[PCstr,[', FR = ' num2str(fr)]]},'fontweight','normal','fontsize',8); set(gca,'xtick',[],'ytick',[])
line([0 trial_length/dt],[10.5 10.5],'color','k');

end

