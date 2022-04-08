function [pc,fr] = plotParamvsPerf_1D(results,nVaries)

nSims = length(results)/nVaries/20;

for ns = 1:nSims
    for n = 1:nVaries
        
        subData = results( ((ns-1)*nVaries + n) : nVaries*nSims : end);
        
        raster = zeros(20,35000);
        
        for t = 1:20
            raster(t,:) = subData(t).C_V_spikes;
        end
        
        [pc.SPIKE(ns,n),pc.ISI(ns,n),pc.RISPIKE(ns,n),pc.spkct(ns,n),fr(ns,n)] = calcPCandFR(raster,20);
        
        
    end
end

end


function [pc_SPIKE,pc_ISI,pc_RISPIKE,pc_spkct,fr] = calcPCandFR(raster,numTrials)

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:))/10000; end
spkTime = reshape(spkTime,numTrials/2,2);

input = reshape(spkTime,1,numTrials);
STS = SpikeTrainSet(input,250/1000,(250+2986)/1000);

distMat = STS.SPIKEdistanceMatrix(250/1000,(250+2986)/1000);
pc_SPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.ISIdistanceMatrix(250/1000,(250+2986)/1000);
pc_ISI = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.RateIndependentSPIKEdistanceMatrix(250/1000,(250+2986)/1000);
pc_RISPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = calcSpkCtDist(spkTime);
pc_spkct = calcpcStatic(distMat, numTrials/2, 2, 0);

fr = mean(sum(raster(:,2500:32500),2))/3;

end



