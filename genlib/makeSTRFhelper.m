% Calculates the correct FFT length, converts stimulus waveform and
% spikeTimes cell arrays to correct size matrices

function [stims,resps,fvec] = makeSTRFhelper(stims,spikeTimes,NBAND,fsSTRF,flims,matlabRmsTo72dBSPL)

if(nargin<6)
    matlabRmsTo72dBSPL = 0.01;
end

[nTrials nstim] = size(spikeTimes); %#ok<ASGLU>
resps = cell(nstim,1);
fs = 0;
if(ischar(stims))
    stimPath = stims;
    stims = cell(nstim,1);
    for stimNum = 1:nstim
        [stims{stimNum},fsCheck] = wavread(fullfile(stimPath,sprintf('stim_%i.wav',stimNum)));
        if(fs ~= fsCheck && stimNum > 1)
            error('Sample rate mismatch between stim_1.wav (%i Hz) and stim_%i.wav (%i Hz) in:\n%s',fs,stimNum,fsCheck,fullfile(stimPath,sprintf('stim_%i.wav',stimNum)));
        end
        fs = fsCheck;
    end
else
    if(length(stims) ~= nstim+1)
        error('Number of stimuli in cell array and number of responses in spikeMatrix must match.');
    end
    fs = stims{end};
    if(isempty(fs) || fs <= 0)
        error('Sample rate FS must be a non-empty positive scalar');
    end
    stims = stims(1:nstim);
    if size(stims,1) == 1
        stims = stims.';
    end
end
if(flims(1)<0);            flims(1)=0; warning('Setting FLIMS(1)=0, the minimum possible frequency.'); end
if(flims(2)>floor(fs/2));  flims(2)=floor(fs/2); warning(sprintf('Setting FLIMS(2)=FS/2=%i, the maximum possible frequency.',fs/2)); end
if(flims(1)>=flims(2));    error('FLIMS(1) must be less than FLIMS(2)'); end;
    
% Find the FFT length that covers the specified rang, coming
% closest to the lower bound. If NBAND=64, this means we want 64
% bands in the range [250,8014], for example.

freqsIn = 0;
bestFftLen = 0;
fftLen = 1;
smallestDiff = fs;
while (sum(freqsIn) <= NBAND && fftLen <= 16384)
    freqs = (0:fftLen/2)*(fs/fftLen);
    freqsIn = freqs>=flims(1) & freqs<=flims(2);
    if(sum(freqsIn) == NBAND)
        thisDiff = freqs(find(freqsIn,1,'first'))-flims(1);
        if(thisDiff <= smallestDiff)
            bestFftLen = fftLen;
            smallestDiff = thisDiff;
        end
    end
    fftLen = fftLen + 1;
end
if(fftLen > 16384)
    error('Could not calculate the appropriate fft length');
end
fftLen = bestFftLen;
freqs = (0:fftLen/2)*fs/fftLen;
fInds = find(freqs>=flims(1) & freqs<=flims(2));
fvec = (fInds-1)*(fs/fftLen);

% A 0.01rms stimulus in matlab corresponds to 72dBSPL playback.
% A 0.01rms stimulus will produce an fft also with 0.01rms.
% Since 20*log10(0.01) = -40dB, 72dbSPL corresponds to -40dB here. Since
% we consider our noise floor 20dBSPL, we want to make 20dBSPL=0 for 
% analysis. Thus we shift everything by (72-(-40)-20) = 92dB, and zero 
% anything below that.
stimShiftDB = 72-20*log10(matlabRmsTo72dBSPL)-20;

% Process stimuli to make spectrograms
for stimNum = 1:nstim
    stimDur = length(stims{stimNum})/fs;
    respLen = ceil(stimDur*fsSTRF);
    stimPad = zeros(ceil(fftLen/2),1);
    stim = [stimPad;stims{stimNum}(:);stimPad;stimPad]; %#ok<AGROW>
    fftOffsets = round(linspace(1,respLen/fsSTRF*fs,respLen));
    spec = zeros(NBAND,respLen,'double');
    myWin = hann(fftLen);
    for ni = 1:respLen
        % Calculate the appropriate fft
        temp = fft(myWin.*stim(fftOffsets(ni):fftOffsets(ni)+fftLen-1));
        spec(:,ni) = 20*log10(abs(temp(fInds)/sqrt(fftLen)));
    end
    
    stims{stimNum} = spec.';
    resps{stimNum} = double(st2sm(spikeTimes(:,stimNum),fsSTRF,stimDur)); %#ok<NASGU>
    if(size(resps{stimNum},1) ~= respLen)
        error('Spike matrix and spectrogram are not the same length!')
    end
    stims{stimNum} = max(stims{stimNum}+stimShiftDB,0);
end
