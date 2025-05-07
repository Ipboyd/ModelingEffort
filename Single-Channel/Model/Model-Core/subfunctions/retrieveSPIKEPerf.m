function [pc] = retrieveSPIKEPerf(results,nVaries)

% do this for full grids only

for n = 1:nVaries

    subData = results( n : nVaries : end );
    for z = 1:24

        raster = zeros(20,35000);

        for t = 1:20
            raster(t,:) = subData(t).C_V_spikes(35000*(z-1)+1:35000*z);
        end

        [pc(n,z)] = calcPCandFR(raster,20);

    end

end

end


function [pc] = calcPCandFR(raster,numTrials)

% spks to spiketimes in a cell array of 20x2
spkTime = cell(numTrials,1);
for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:))/10000; end
spkTime = reshape(spkTime,numTrials/2,2);

input = reshape(spkTime,1,numTrials);
STS = SpikeTrainSet(input,250/1000,(250+3000)/1000);

distMat = STS.SPIKEdistanceMatrix(250/1000,(250+3000)/1000);
pc = calcpcStatic(distMat, numTrials/2, 2, 0);

end
