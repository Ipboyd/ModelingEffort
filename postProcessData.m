function [perf,fr] = postProcessData(data,options)
% calculate performance and FR
% no plotting function for now

time_end = options.time_end;
plot_rasters = options.plotRasters;
numTrials = length(data);

jump = length(find([data.Inh_Inh_trial]==1));

for vv = 1:jump % for each varied parameter
    subData = data(vv:jump:length(data));

    % visualize spikes
    ICspks = zeros(numTrials,4,time_end);
    Rspks = zeros(numTrials,4,time_end);
    Cspks = zeros(numTrials,time_end);
    for i = 1:numTrials
        for j = 1:4
            ICspks(i,j,:) = subData(i).Exc_V_spikes(:,j);
            Rspks(i,j,:) = subData(i).R_V_spikes(:,j);
        end
        Cspks(i,:) = subData(i).C_V_spikes;
    end
    [perf.C(vv),fr.C(vv)] = calcPCandPlot(Cspks,time_end,1,plot_rasters,numTrials);     
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