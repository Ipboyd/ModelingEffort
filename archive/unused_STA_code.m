
% %% Spike-triggered averages
% % close all;
% 
% pops = {'R2On'}; %{'On','R1On','R2On'};
% clearvars spks
% 
% load('AM_sigs.mat');
% % load('bandlimited_stim.mat');
% AM_sigs = cell2mat(AM_sig)';
% 
% fs_stim = 20000;
% tlen_STA = .150; % [seconds]
% 
% figure;
% for p = 1:length(pops)
%     spks = sprintf('%s_V_spikes',pops{p});
%     for a = 1:5 % for each target
%         subplot(5,1,a);
%         temp = [];
%         % get spiketimes for each trial
%         for t = (1:10)+a*10 %a*10% (1:10)+(a-1)*10
%             temp = [temp; find(snn_out(t).(spks))/10000];
%         end
% 
%         % for each spike time, get last 150 ms of target stimulus
% 
%         % convert to indexes from target
%         inds = round(temp * fs_stim); inds(inds < round(fs_stim*tlen_STA) | inds > size(AM_sigs,2)) = [];
% 
%         t_STA = (-round(fs_stim*tlen_STA):0)/fs_stim;
%         inds = (-round(fs_stim*tlen_STA):0) + inds;
%         STA = [];
%         for i = 1:size(inds,1)
%             STA = cat(1,STA,AM_sigs(a,inds(i,:)));
%         end
%         STA_mean = mean(STA);
% 
%         plot(t_STA,STA_mean,'linewidth',1); hold on;
% 
%         ylabel(sprintf('%i Hz',AM_freqs(a)));
% 
%     end
% end
% sgtitle('Direct STA');
% 
% savefig(gcf,[simDataDir filesep 'direct STA.fig']);
% % close;

% %% STA from envelope
% 
% AM_freqs = [2 4 8 16 32];
% fs_stim = 20000;
% tlen_STA = .150; % [seconds]
% 
% % for bandlimited stimuli
% for a = 1:2
% y = envelope(AM_sigs(a,:),10000);
% dbsig = 20*log10(abs(y));
% end
% 
% clearvars envelope envelope_db stim_env stim_env_db
% for a = 1:2%length(AM_freqs)
%     %envelope(a,:) = [ zeros(round(fs_stim*0.25),1); (0.5*sin(2*pi*AM_freqs(a)*(0:round(fs_stim*1.5))'/fs_stim)+1) ];
%     %envelope_db(a,:) = [ zeros(round(fs_stim*0.25),1); 20*log10(0.5*sin(2*pi*AM_freqs(a)*(0:round(fs_stim*1.5))'/fs_stim)+1) ];
%     stim_env(a,:) = envelope(AM_sigs(a,:),10000); stim_env(a,stim_env(a,:)<0) = 1E-06;
%     stim_env_db(a,:) = 20*log10(stim_env(a,:));
% end
% 
% pops = {'R2On'}; %{'On','R1On','R2On'};
% clearvars spks
% 
% % load('AM_sigs.mat');
% 
% figure;
% for p = 1:length(pops)
%     spks = sprintf('%s_V_spikes',pops{p});
%     for a = 1:2%5 % for each target
%         subplot(5,1,a);
%         temp = [];
%         % get spiketimes for each trial
%         for t = (1:10)+(a-1)*10
%             temp = [temp; find(snn_out(t).(spks))/10000];
%         end
% 
%         % for each spike time, get last 150 ms of target stimulus
% 
%         % convert to indexes from target
%         inds = round(temp * fs_stim); inds(inds < round(fs_stim*tlen_STA) | inds > size(stim_env,2)) = [];
% 
%         t_STA = (-round(fs_stim*tlen_STA):0)/fs_stim;
%         inds = (-round(fs_stim*tlen_STA):0) + inds;
%         STA = []; STA_dB=[];
%         for i = 1:size(inds,1)
%             STA = cat(1,STA,stim_env(a,inds(i,:)));
%             STA_dB = cat(1,STA_dB,stim_env_db(a,inds(i,:)));
%         end
%         STA_mean = mean(STA);
%         STA_dB_mean = mean(STA_dB);
% 
%         yyaxis left;
%         plot(t_STA,STA_mean,'linewidth',1); ylabel(sprintf('%i Hz',AM_freqs(a)));
%         yyaxis right;
%         plot(t_STA,STA_dB_mean,'linewidth',1);
%     end
% end
% sgtitle('Envelope STA');
% legend('Linear','dB');
% 
% savefig(gcf,[simDataDir filesep 'envelope STA.fig']);