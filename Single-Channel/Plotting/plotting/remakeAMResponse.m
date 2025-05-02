close all;

TrialsPerStim = 20;
nVaries = 1;
nStim = 5;
dt = 0.1;
simDataDir = fullfile('simData-AM','no offset PV');
mkdir(simDataDir);

t_stim = 3;
padToTime = (t_stim + 0.250 + 0.250) * 1000;

clearvars RM

for nV = 1:nVaries

    figure(1);

    AM_freqs = [2 4 8 16 32];

    t_bin = 20; %ms
    t_vec = 0:t_bin:(padToTime-1);
    for a = 1:nStim
        figure(1);
        subplot(nStim,1,a);

        % convert peakDrv to ms
        temp = [];
        % get spiketime indexes for each trial
        for t = ((0:TrialsPerStim-1)*nVaries + nV) + ((a-1)*10*nVaries) % (1:TrialsPerStim)+(a-1)*10
            temp = [temp; find(results(t).R2On_V_spikes)];
        end

        % convert to PSTH
        PSTH = histcounts(temp,0:t_bin*10:padToTime*10);
        plot(t_vec,PSTH,'k','linewidth',1); hold on;
        plot([peakDrv{a};peakDrv{a}],[0 20]'.*ones(size(peakDrv{a})),'--r','linewidth',1);
        ylabel(sprintf('%i Hz',AM_freqs(a)))
        meanFR = mean(PSTH(t_vec >= 250 & t_vec < t_stim*1000+250)) * (1000/t_bin) / TrialsPerStim;
        title_str = sprintf('Mean FR during AM stim: %.0f Hz',meanFR);
        title(title_str,'fontweight','normal');

        % get activity within 150ms of each peakDrv event
        % convert spike time indices to ms
        peakAct = [];
        for d = 1:length(peakDrv{a})
            if peakDrv{a}(d) ~= 250
                temp2 = temp*dt - peakDrv{a}(d);
                peak_vec = -200 : 5 : 200;
                peakAct = cat(1,peakAct,histcounts(temp2,peak_vec));
            end
        end
        figure(2);
        subplot(nStim,1,a);
        plot(peak_vec(1:end-1),mean(peakAct)); hold on;
        plot([0 0],[0 10],'r');
        ylabel(sprintf('%i Hz',AM_freqs(a)));
    end

    figure(1);
    xlabel('Time (ms)');
    savefig(gcf,fullfile(simDataDir,filesep,sprintf('R2On_AMresponse, vary %i.fig',nV)));

    figure(2);
    xlabel('Time from peakDrv (ms)');
    savefig(gcf,fullfile(simDataDir,filesep,sprintf('R2On_peakDrv, vary %i.fig',nV)));

    %% Look at peakDrv response across excitatory layers
    % close all;
    figure('unit','inches','position',[4 4 1.3 3])

    t_bin = 20; %ms
    t_vec = 0:t_bin:(padToTime-1);
    pops = {'On','R1On','R2On'};
    for p = 1:length(pops)
        spks = sprintf('%s_V_spikes',pops{p});
        for a = 1:nStim
            subplot(nStim,1,a);

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
            plot(peak_vec(1:end-1),mean(peakAct)); hold on;
            if p == length(pops), plot([0 0],[0 15],'r'); end
            ylabel(sprintf('%i Hz',AM_freqs(a)));
            box off;

            xlim([-200 200]);
            ylim([0 15]);
            set(gca,'fontsize',8,'xtick',-200:50:200,'ytick',[0 15]);
            if a == 5
                set(gca,'xticklabels',{-200,[],[],[],0,[],[],[],200},'xticklabelrotation',0);
            else
                set(gca,'xticklabels',[]);
            end

            % calculate rate modulation per peakAct event
            meanAct = mean(peakAct);
            minFR = min(meanAct); maxFR = max(meanAct);
            RM(nV,a).(pops{p}) = (maxFR-minFR)/(minFR+maxFR);
        end
    end
    xlabel('Time (ms)');
    legend(pops);
    savefig(gcf,fullfile(simDataDir,filesep,sprintf('exc peakDrv responses, vary %i.fig',nV)));
    close all;

end