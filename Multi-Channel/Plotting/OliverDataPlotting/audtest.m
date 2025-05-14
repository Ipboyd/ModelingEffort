% 
% [audioIn, fs] = audioread('200k_target1.wav');
% vad = voiceActivityDetector()
% 
% isSpeech = vad(audioIn);
% 
% diffSpeech = diff([0; isSpeech; 0]);
% onsets = find(diffSpeech == 1);
% offsets = find(diffSpeech == -1) - 1;
% 
% % Convert to time (seconds)
% 
% onsetTimes = onsets / fs;
% offsetTimes = offsets / fs;

afr = dsp.AudioFileReader('200k_target1.wav');
fs = afr.SampleRate;

frameSize = ceil(5e-3*fs);
overlapSize = ceil(.*frameSize);
hopSize = frameSize - overlapSize;
afr.SamplesPerFrame = hopSize;

inputBuffer = dsp.AsyncBuffer('Capacity',frameSize);

VAD = voiceActivityDetector('FFTLength',650000);

scope = timescope('SampleRate',fs, ...
    'TimeSpanSource','Property','TimeSpan',3, ...
    'BufferLength',.5*fs, ...
    'YLimits',[-1.5,1.5], ...
    'TimeSpanOverrunAction','Scroll', ...
    'ShowLegend',true, ...
    'ChannelNames',{'Audio','Probability of speech presence'});

player = audioDeviceWriter('SampleRate',fs);

pHold = ones(hopSize,1);

while ~isDone(afr)
    x = afr();
    n = write(inputBuffer,x);

    overlappedInput = read(inputBuffer,frameSize,overlapSize);

    p = VAD(overlappedInput);

    pHold(end) = p;
    scope(x,pHold)

    player(x);

    pHold(:) = p;
end