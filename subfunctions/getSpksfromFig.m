function [raster,pc,fr] = getSpksfromFig(pop)

all_plots = findobj(gcf,'type','axes');

for a = 1:length(all_plots)
    pop_title{a} = all_plots(a).Title.String{1};
end

ax_plot = all_plots(strcmp(pop_title,pop));

all_lines = findobj(ax_plot,'type','line');

spk_times = all_lines(2).XData; spk_times(2:3:end) = [];
spk_trials = all_lines(2).YData; spk_trials(2:3:end) = [];

raster = zeros(20,35000);
for t = 1:20
    times = spk_times(spk_trials == t-0.5);
    raster(t,times) = 1;
end

[pc,fr] = calcPCandFR(raster,20);

end

function [pc,fr] = calcPCandFR(raster,numTrials)

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
spkTime = reshape(spkTime,numTrials/2,2);

input = reshape(spkTime,1,numTrials);
STS = SpikeTrainSet(input,250*10,(250+2986)*10);
distMat = STS.SPIKEdistanceMatrix(250*10,(250+2986)*10);
pc.SPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.RateIndependentSPIKEdistanceMatrix(250*10,(250+2986)*10);
pc.RISPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.ISIdistanceMatrix(250*10,(250+2986)*10);
pc.ISI = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = calcSpkCtDist(spkTime);
pc.spk_ct = calcpcStatic(distMat, numTrials/2, 2, 0);

fr = mean(sum(raster(:,2500:32500),2))/3;

end


