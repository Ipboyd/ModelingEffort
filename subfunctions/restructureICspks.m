function restructureICspks(ICdir)
% restructure IC spikes from 10x8 cell to trial x location x time

ICFiles = dir([ICdir '*.mat']);
mkdir([ICdir filesep 'spikeTrains']);

for i = 1:length(ICFiles)
    load([ICdir ICFiles(i).name],'t_spiketimes');
    temp = cellfun(@max,t_spiketimes,'UniformOutput',false);
    tmax = max([temp{:}]);
    spks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
    for j = 1:size(t_spiketimes,1) %trials [1:10]
        for k = 1:size(t_spiketimes,2) %neurons [(1:4),(1:4)]
            if k < 5 %song 1
                spks(j,k,round(t_spiketimes{j,k})) = 1;
            else
                spks(j+10,k-4,round(t_spiketimes{j,k})) = 1;
            end
        end
    end
    
    save(fullfile(ICdir,'spikeTrains',ICFiles(i).name),'spks');
end