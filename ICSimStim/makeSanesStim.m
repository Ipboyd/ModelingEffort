
AM_freqs = [2 4 8 16 32];

fs = 20000;

t = 1.5;
targetlvl = 0.01;

carrier = randn(round(t*fs),1);
carrier = bandpass(carrier,[100 10000],fs);

% 1st row of AM_sigs is unmodulated white noise
AM_sig = carrier;
AM_sig = AM_sig/rms(AM_sig)*targetlvl;

t_vec = (0:(length(carrier)-1))/fs;
for a = 1:length(AM_freqs)
    env = -cos(2*pi*AM_freqs(a)*t_vec')+1;
    AM_sig = carrier.*env;

    AM_sig = AM_sig/rms(AM_sig)*targetlvl;
    
    % get envelope using hilbert, convert to db
    dbsig = 20*log10(abs(env));

    % find peakDrv events (in ms)
    peakDrv{a} = 1000*t_vec(islocalmin(dbsig)); %peakDrv{a}(1) = [];
    % peakDrv_lin{a} = 1000*t_vec(islocalmax(diff(env))); %peakDrv{a}(1) = [];
end
save('peakDrv_AM.mat','peakDrv')

%% 

clearvars x y dbsig envelope

[y1,fs] = audioread('200k_target1.wav');
[y2,fs] = audioread('200k_target2.wav');

x{1} = y1;
x{2} = y2;

separation = round(fs*0.02); % minimum peak separation by 20ms

for t = 1:2

y = smooth(envelope(x{t},10000),10000);
dbsig = 20*log10(abs(y));

t_vec = (0:length(x{t})-1) / fs;

dbmin = islocalmin(dbsig,'MinSeparation',separation,'MinProminence',1);
dbmax = islocalmax(dbsig,'MinSeparation',separation,'MinProminence',1);

min_inds = find(dbmin); max_inds = find(dbmax);
peakDrv_target{t} = []; peakDrv_inds =[];
for d = 1:length(min_inds)
    % find ensuing max
    temp = max_inds(find(max_inds - min_inds(d) > 0,1));
    % peakDrv: only where ensuing dbmax is higher than 6 dB above dbmin
    if dbsig(temp) - dbsig(min_inds(d)) > 6
        peakDrv_target{t} = [peakDrv_target{t}, t_vec(min_inds(d))*1000];
        peakDrv_inds = [peakDrv_inds, min_inds(d)];
    end
end

y_env{t} = y;

figure; plot(t_vec,dbsig); hold on;
plot(t_vec(min_inds),dbsig(min_inds),'r*');
plot(t_vec(max_inds),dbsig(max_inds),'g*');

figure; plot(t_vec,dbsig); hold on; plot(peakDrv_target{t} / 1000,dbsig(peakDrv_inds),'r*');
end

save('target_envelopes.mat','y_env','fs');