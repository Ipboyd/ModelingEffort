function [perf,fr] = postProcessData_new(data,s,trialStart,trialEnd,configName,options)
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
plot_rasters = options.plotRasters;
jump = length(find([data.Inh_Inh_trial]==1));
numTrials = length(data)/jump; %usually, 20 trials
numChannels = 4;

% network properties
popNames = {s.populations.name};
popSizes = [data(1).model.specification.populations.size];
nPops = numel(popNames);
fieldNames = strcat(popNames,'_V_spikes');

% for this trial
tstart = trialStart;
tend = trialEnd;


% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end
if plot_rasters, figure; end
for vv = 1:jump % for each varied parameter
    subData = data(vv:jump:length(data)); %grab data for this param variation
    variedParamVal = mode([subData.(options.variedField)]); % should all be the same
    for currentPop = 1:nPops
        
        % skip processing for current population if not within specified subpopulation.
        if ~contains(popNames(currentPop),options.subPops), continue; end
        
        % for each spatial channel
        for channelNum = 1:popSizes(currentPop) 
            channel = struct();
            
            % for each trial
            for trial = 1:numTrials 
                channel(channelNum).popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(tstart:tend,channelNum);
            end
            figName = sprintf('%s_%s%.02f_%s_channel%i',configName,options.variedField,variedParamVal,popNames{currentPop},channelNum);
            [perf.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv),...
             fr.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv)] = ...
             calcPCandPlot(channel(channelNum).popSpks,time_end,1,numTrials,plot_rasters,figName);
        end
    end
end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,plot_rasters,figName)

PCstr = '';

if calcPC
    % spks to spiketimes in a cell array of 20x2
    tau = linspace(1,30,100);
    spkTime = cell(numTrials,1);
    for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
    spkTime = reshape(spkTime,numTrials/2,2);
    % calculate distance matrix & performance
    distMat = calcvr(spkTime, tau);
    [performance, ~] = calcpc(distMat, numTrials/2, 2, 1,[], 'new');
    pc = mean(max(performance));
    PCstr = ['PC = ' num2str(pc)];
end

if size(raster,2) > 3000
fr = 1000*mean(sum(raster(:,[1:1000,4000:end]),2))/2000;
else
    fr = 1000*mean(sum(raster,2))/3000;
end
%plot
if plot_rasters
    clf
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    title({PCstr,['FR = ' num2str(fr)]});
    xlim([0 time_end])
    line([0,time_end],[numTrials/2 + 0.5,numTrials/2 + 0.5],'color',[0.3 0.3 0.3])
    saveas(gcf,[figName '.tif'])
end
    
end