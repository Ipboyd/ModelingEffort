function [pc,fr] = calcPerfsandFR(varargin)

% calculates performance not only based on SPIKE-distance (already
% calculated in postProcessData_new), but also the other Kreuz distances
% (ISI, RI-SPIKE) and spike count distance

spks = varargin{1};
nVaried = varargin{2}; % how much parameter sets are there, excluding the trials and number of repeat trials for laser
dt = varargin{3}; 
pop = varargin{4};

nSims = length(spks.(pop)) / nVaried; % how many repeats (set of 20 trials) for each parameter set (should be 5 for laser and control)

chanNames = fieldnames(spks.(pop));
nChans = length(chanNames);

for vv = 1:nVaried
    for ns = 1:nSims

        % fetch 20 trials to calculate performance at specific spot
        popSpks = spks.(pop)(vv);

        for ch = 1:nChans
            raster = popSpks.(chanNames{ch});
            numTrials = size(raster,1);

            [pc.SPIKE(ns,vv,ch),pc.ISI(ns,vv,ch),pc.RISPIKE(ns,vv,ch),pc.spkct(ns,vv),fr(ns,vv,ch)] = calcPCandFR(raster,numTrials,dt);
        end

    end
end

end


function [pc_SPIKE,pc_ISI,pc_RISPIKE,pc_spkct,fr] = calcPCandFR(raster,numTrials,dt)

start_time = 300; % [ms]
end_time = start_time + 3000; % [ms];

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials
    spkTime{ii} = find(raster(ii,:))*dt;
end
spkTime = reshape(spkTime,numTrials/2,2);
input = reshape(spkTime,1,numTrials);
fr = round(mean(cellfun(@(x) sum(x >= start_time & x < end_time) / 3,input)));

STS = SpikeTrainSet(input,start_time,end_time);

distMat = STS.SPIKEdistanceMatrix(start_time,end_time);
pc_SPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.ISIdistanceMatrix(start_time,end_time);
pc_ISI = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = STS.RateIndependentSPIKEdistanceMatrix(start_time,end_time);
pc_RISPIKE = calcpcStatic(distMat, numTrials/2, 2, 0);

distMat = calcSpkCtDist(spkTime,start_time,end_time);
pc_spkct = calcpcStatic(distMat, numTrials/2, 2, 0);

end
