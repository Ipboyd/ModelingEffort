function [perf,fr] = postProcessData_new(data,s,options)
% calculate performance and FR
% no plotting function for now

time_end = options.time_end;
plot_rasters = options.plotRasters;
numTrials = length(data);
numChannels = 4;

% network properties
popNames = {s.populations.name};
popSizes = [data(1).model.specification.populations.size];
nPops = numel(popNames);
fieldNames = strcat(popNames,'_V_spikes');

% for this trial
% tstart = options.trialStart;
% tend = options.trialEnd;

jump = length(find([data.Inh_Inh_trial]==1));

if ~isfield(options,'subPops'), options.subPops = popNames; end
for vv = 1:jump % for each varied parameter
    subData = data(vv:jump:length(data));
    % visualize spikes
    for currentPop = 1:nPops
        % skip processing for current population if not within specified subpopulation.
        if ~contains(popNames(currentPop),options.subPops), continue; end
        popSpks = zeros(numTrials,popSizes(currentPop),time_end);
        for channelNum = 1:popSizes(currentPop) % for each spatial channel
            channel = struct();
            for trial = 1:numTrials % for each trial
                channel(channelNum).popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(:,channelNum);
            end
            [perf.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv),...
             fr.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv)] = ...
             calcPCandPlot(channel(channelNum).popSpks,time_end,1,plot_rasters,numTrials);
        end
    end
end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,plot_rasters,numTrials)

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
    figure;
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    title({PCstr,['FR = ' num2str(fr)]});
    xlim([0 time_end])
    line([0,time_end],[numTrials/2 + 0.5,numTrials/2 + 0.5],'color',[0.3 0.3 0.3])
end
    
end