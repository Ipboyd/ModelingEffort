
nVaries = length(snn_out)/nTrials;

% calculate performance
data = struct();
options.time_end = padToTime; %ms
options.variedField = strrep(expVar,'-','_');

annotTable = createSimNotes(snn_out,simDataDir,options);

% save C spikes and varied params to struct
names = snn_out(1).varied;

pops = {snn_out(1).model.specification.populations.name};
results = struct;
for i = 1:length(snn_out)
    % results(i).R2On_V_spikes = snn_out(i).R2On_V_spikes;
    for p = 1:length(pops)
    results(i).([pops{p} '_V_spikes']) = snn_out(i).([pops{p} '_V_spikes']);
    end
    for t = 1:length(names)
        results(i).(names{t}) = snn_out(i).(names{t});
    end
end
results(1).model = snn_out(1).model; save([simDataDir filesep 'spikes.mat'],'results');

%% convert peakDrv to samples (10000 Hz)

close all;

figure;

for nV = 1:nVaries

    subplot(2,1,1);

    t_bin = 20;
    t_vec = 0:t_bin:padToTime; t_vec(end) = [];

    % convert peakDrv to ms
    temp = [];
    % get spiketime indexes for each trial
    for t = ((0:TrialsPerStim-1)*nVaries + nV)
        temp = [temp; find(snn_out(t).R2On_V_spikes)];
    end
    PSTH = histcounts(temp,0:t_bin*10:padToTime*10);
    plot(t_vec,PSTH,'k','linewidth',1); hold on;

    meanFR = mean(PSTH(t_vec >= 250 & t_vec < t_stim*1000+250)) * (1000/t_bin) / TrialsPerStim;
    title_str = sprintf('Mean FR: %.0f Hz',meanFR);
    title(title_str,'fontweight','normal');

    % get activity within 150ms of each peakDrv event
    % convert spike time indices to ms
    peakAct = [];
    for d = 1:length(peakDrv)
        temp2 = temp*dt - peakDrv(d);
        peak_vec = -150 : 6 : 150;
        peakAct = cat(1,peakAct,histcounts(temp2,peak_vec));
    end
    subplot(2,1,2);
    plot(peak_vec(1:end-1),mean(peakAct)); hold on;
    plot([0 0],[0 10],'r');

    xlabel('Time from peakDrv (ms)');
    savefig(gcf,fullfile(simDataDir,filesep,sprintf('R2On response, vary %i.fig',nV)));

    % Look at peakDrv response across excitatory layers
    % close all;
    figure('unit','inches','position',[4 4 3 2])

    t_bin = 20;
    t_vec = 0:t_bin:(padToTime-1);
    pops = {'On','R1On','R2On'};
    for p = 1:length(pops)
        spks = sprintf('%s_V_spikes',pops{p});

        temp = [];
        % get spiketime indexes for each trial
        for t = ((0:TrialsPerStim-1)*nVaries + nV)
            temp = [temp; find(snn_out(t).(spks))];
        end

        % get activity within 150ms of each peakDrv event
        % convert spiketime indices to ms
        peakAct = [];
        for d = 1:length(peakDrv)
            temp2 = temp*dt - peakDrv(d);
            peak_vec = -150 : 6 : 150;
            peakAct = cat(1,peakAct,histcounts(temp2,peak_vec));
        end
        plot(peak_vec(1:end-1),mean(peakAct)); hold on;
        if p == length(pops), plot([0 0],[0 10],'r'); end
    end
    ylabel('FR (Hz)');
    xlabel('Time (ms)');
    legend(pops);
    savefig(gcf,fullfile(simDataDir,filesep,sprintf('exc peakDrv responses, vary %i.fig',nV)));
    %close all;
end

