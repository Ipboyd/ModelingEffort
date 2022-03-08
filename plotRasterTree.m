function [perf,fr,spks] = plotRasterTree(data,s,tstart,tend,configName,options)
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

time_end = options.time_end;
jump = length(find([data.On_On_trial]==1));
numTrials = length(data)/jump; % # total trials (20 for single parameter set)

% network properties
popNames = {s.populations.name};
nPops = numel(popNames); 
fieldNames = strcat(popNames,'_V_spikes');

% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end

% in same order as popNames
if nPops > 6 % on and off columns
    subplot_locs = [14,9,10,5,6,16,11,12,7,8];
else % one column only
    subplot_locs = [14,9,10,5,6];
end
% locs = {'90','45','0','-90'};

for vv = 1:jump % for each varied parameter
    figure('unit','inches','position',[6 3 7.5 5]);
    subData = data(vv:jump:length(data)); %grab data for this param variation
    
    try
        variedParamVal = mode([subData.(options.variedField)]); % should all be the same
    catch
        variedParamVal = 0;
    end
    
    for currentPop = 1:nPops
        
        channel = struct();
        
        % for each trial
        for trial = 1:numTrials
            channel.popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(tstart:tend);
        end
        spks.(popNames{currentPop})(vv).channel = channel.popSpks;
        
        [perf.(popNames{currentPop}).channel(vv),...
            fr.(popNames{currentPop}).channel(vv)] = ...
            calcPCandPlot(channel.popSpks,time_end,1,numTrials,popNames{currentPop},...
            subplot_locs(currentPop));
        
    end
    
    figName = sprintf('%s_%s%.03f_set%s',configName,options.variedField,...
        variedParamVal,num2str(vv));
    
    annotation('textbox',[.1 .82 .1 .1], ...
        'String',configName(end-3:end),'EdgeColor','none','FontSize',20)

    saveas(gcf,[figName '.png']);
    
end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,unit,subplot_loc)

PCstr = '';

if calcPC
    % spks to spiketimes in a cell array of 20x2
    spkTime = cell(numTrials,1);
    for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
    spkTime = reshape(spkTime,numTrials/2,2);
    
    input = reshape(spkTime,1,numTrials);
    STS = SpikeTrainSet(input,250*10,(250+2986)*10);
    distMat = STS.SPIKEdistanceMatrix(250*10,(250+2986)*10);
    
    performance = calcpcStatic(distMat, numTrials/2, 2, 0);
    pc = mean(max(performance));
    PCstr = ['PC = ' num2str(round(pc))];
end

fr = round(1000*mean(sum(raster(:,[2500:32500]),2))/3000);

[r,c] = ind2sub([4 4],subplot_loc);
r = r; 
c = 5-c;

xpos = 0.06 + 0.23*(r-1);
ypos = 0.06 + 0.25*(c-1);
x = 0.19;
y = 0.12;

subplot('position',[xpos ypos x y]);

raster = raster(1:10,:);

% plot only target 1 response
plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
xlim([0 time_end*10]); ylim([0.5 10.5]);
title({unit,PCstr,['FR = ' num2str(fr)]}); set(gca,'xtick',[],'ytick',[])

end
