
TrialsPerStim = 20;
nVaries = 1;
nV = 1;
dt = 0.1;

freqs = [2 4 8 16 32];
clearvars RM
pops = {'R2On'};
for p = 1:length(pops)
    spks = sprintf('%s_V_spikes',pops{p});
    for a = 1:5

        temp = []; % contains indexes for all trials
        % get spiketime indexes for each trial
        for t = ((0:TrialsPerStim-1)*nVaries + nV) + ((a-1)*10*nVaries)
            temp = [temp; find(results(t).(spks))];
        end

        % get activity within 200ms of each peakDrv event
        peakAct = [];
        for d = 1:length(peakDrv{a})
            if peakDrv{a}(d) ~= 250

                % convert spiketime indices to ms
                temp2 = temp*dt - peakDrv{a}(d);
                peak_vec = -200 : 5 : 200;
                peakAct = cat(1,peakAct,histcounts(temp2,peak_vec));
            end
        end

        meanAct = mean(peakAct);
        minFR = min(meanAct); maxFR = max(meanAct);
        % RM(nV,a).(pops{p}) = (maxFR-minFR)/(minFR+maxFR);
        RM(a) = (maxFR-minFR)/(minFR+maxFR);

    end

    RM = RM/RM(1);

    ind = find(RM < 0.5,1);

    tempfreqs = linspace(freqs(ind-1),freqs(ind),400);

    m = (RM(ind)-RM(ind-1))/(freqs(ind)-freqs(ind-1));
    freq_RM = (0.5 - RM(ind-1))/m + freqs(ind-1)

end
