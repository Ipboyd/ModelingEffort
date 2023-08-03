PSTHs = zeros(6,20000);
for a = 1:60
PSTHs(ceil(a/10),:) = results(a).R2On_V_spikes' + PSTHs(ceil(a/10),:); end


% tuning curve
FR = sum(PSTHs_new,2)/10/2;
figure; plot([0 2 4 8 16 32],FR);
ylim([0 40]); ylabel('FR (Hz)'); xlabel('AM frequency (Hz)');

% figure;
% for a = 1:6
%     inds = ((a-1)*10+1):10*a;
%     spks = [];
%     for i = inds
%     spks = cat(2,spks,results(i).R2On_V_spikes);
%     end
%     plot(spks' + [0:9]);
% end


%% Plot PSTH vs AM freq
AM_freqs = [0 2 4 8 16 32];

fs = 10000;

t = 1.5;
carrier = randn(round(t*fs),1);

% 1st row of AM_sigs is unmodulated white noise
AM_sig = [];
AM_sig(1,:) = carrier;

t_vec = (0:(length(carrier)-1))/fs;
for a = 1:length(AM_freqs)
    envelope = sin(2*pi*AM_freqs(a)*t_vec');
    AM_sig(a+1,:) = carrier.*envelope;
end


figure; PSTHs_new = [];
kt = 0:1:100;
kern = exp(-kt/5);
for a = 1:6
    subplot(6,1,a)
%     temp = histcounts(find(PSTHs(a,:)),0:10:20000);
%     oldlen = numel(temp);
%     temp = conv(temp,kern);
%     PSTHs_new(a,:) = temp(1:oldlen);
    PSTHs_new(a,:) = histcounts(find(PSTHs(a,:)),0:200:20000);
    plot(PSTHs_new(a,:),'linewidth',1); hold on;
    ylim([0 25]);
    title(sprintf('AM freq = %i Hz',AM_freqs(a)))
end





