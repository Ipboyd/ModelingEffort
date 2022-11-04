function [pc,fr] = getPerffromRaster()

ax = findobj(gcf,'type','axes');

distLabels = {'SPIKE','ISI','RateIndependentSPIKE'};

pc = struct; fr = struct;

for n = 1:length(ax)
    pc.(ax(n).Title.String{1}) = struct;
    % get spike data from rasters
    a = findobj(ax(n),'type','line');
    spk_inds = a(2).XData(2:3:end);
    spk_trials = a(2).YData(2:3:end)-0.5;
    raster = zeros(20,35000);

    for t = 1:20
        raster(t,spk_inds(spk_trials == t)) = 1;
    end

    for d = 1:3
        [pc.(ax(n).Title.String{1}).(distLabels{d}),fr.(ax(n).Title.String{1})] = calcPCandFR(raster,20,distLabels{d});
    end
end

end


function [pc,fr] = calcPCandFR(raster,numTrials,distLabel)

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:))/10000; end
spkTime = reshape(spkTime,numTrials/2,2);

input = reshape(spkTime,1,numTrials);
STS = SpikeTrainSet(input,300/1000,(300+3000)/1000);

distMat = eval(['STS.' distLabel 'distanceMatrix(300/1000,(300+3000)/1000)']);
pc = calcpcStatic(distMat, numTrials/2, 2, 0);

fr = mean(mean(cellfun(@(x) sum(x >= 0.3 & x < 3.3),spkTime))/3);

end
