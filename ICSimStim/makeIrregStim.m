close all
clear
fs = 20000;
targetlvl = 0.01;

%% Band-limited envelope
t = 2;

for n = 1:2

    carrier = randn(round(t*fs),1);
    carrier = bandpass(carrier,[100 9000],fs);

    % envelope

    t_vec = (0:(length(carrier)-1))/fs;

    fp1 = 50; fp2 = 1000;

    env = randn(round(t*fs),1);
    [env_bp,d] = bandpass(env,[fp1 fp2],fs);
    AM_sig{n} = carrier .* (env_bp+min(env_bp));
    AM_sig{n} = AM_sig{n}/rms(AM_sig{n})*targetlvl;

    AM_sig{n} = [zeros(round(fs*0.250),1); AM_sig{n}];

    figure; plot(AM_sig{n});
end
save('bandlimited_stim.mat','AM_sig','fs','fp1','fp2');

%% Random AM rate every cycle

AM_freqs = [2 4 8 16 32];
AM_sig = cell(1,2);

for n = 1:2
    env = [];

    f = AM_freqs(randperm(5,1));

    while length(env)/fs <= 2
        t_vec = (0:1/fs:(1/f - 1/fs))';
        if length([env;t_vec])/fs > 2, break; end
        env = [env;-cos(2*pi*f*t_vec) + 1];
        new_freq = setdiff(AM_freqs,f);
        f = new_freq(randperm(4,1));
    end

    carrier = randn(length(env),1);
    carrier = bandpass(carrier,[100 9000],fs);

    AM_sig{n} = carrier .* env; AM_sig{n} = AM_sig{n} / rms(AM_sig{n}) * targetlvl;
    
    % zeropad 250ms in front
    % AM_sig{n}(end+1:fs*2) = 0;
    AM_sig{n} = [zeros(round(fs*0.250),1); AM_sig{n}];

    figure; plot(AM_sig{n});

    % calculate peakDrv events
    env_db = 20*log10(env);
    peakDrv{n} = 1000*find(islocalmin(env_db))/fs; %ms
end
save('irregular_stim.mat','AM_sig','fs','peakDrv');
