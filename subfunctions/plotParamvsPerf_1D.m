function [pc,fr] = plotParamvsPerf_1D(varargin)

results = varargin{1};
nVaries = varargin{2}; % how much parameter sets are there, excluding the trials and number of repeat trials for laser

if nargin == 3
pop = varargin{3};
else
pop = 'R2On';
end

nSims = length(results)/nVaries/20;

for ns = 1:nSims
    for n = 1:nVaries

        % fetch 20 trials to calculate performance at specific spot
        subData = results( ((ns-1)*nVaries + n) : nVaries*nSims : end);

        nCh = size(subData(1).([pop '_V_spikes']),2);

        for ch = 1:nCh

            raster = zeros(20,35000);

            for t = 1:20
                raster(t,:) = subData(t).([pop '_V_spikes'])(:,ch);
            end

            [pc.SPIKE(ns,n,ch),pc.ISI(ns,n,ch),pc.RISPIKE(ns,n,ch),pc.spkct(ns,n),fr(ns,n,ch)] = calcPCandFR(raster,20);

        end

    end
end

end


function [pc_SPIKE,pc_ISI,pc_RISPIKE,pc_spkct,fr] = calcPCandFR(raster,numTrials)

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:))/10000; end
spkTime = reshape(spkTime,numTrials/2,2);

input = reshape(spkTime,1,numTrials);
STS = SpikeTrainSet(input,300/1000,(300+3000)/1000);

distMat = STS.SPIKEdistanceMatrix(300/1000,(300+3000)/1000);
pc_SPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.ISIdistanceMatrix(300/1000,(300+3000)/1000);
pc_ISI = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.RateIndependentSPIKEdistanceMatrix(300/1000,(300+3000)/1000);
pc_RISPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = calcSpkCtDist(spkTime,0.3,3.3);
pc_spkct = calcpcStatic(distMat, numTrials/2, 2, 0);

fr = mean(sum(raster(:,3001:33000),2))/3;

end
