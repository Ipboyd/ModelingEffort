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
cd(userpath);
cd('../GitHub/ModelingEffort/Single-Channel/Model/PreCortical-Modeling/resampled-stimuli')


% Load target 1
[x, fs] = audioread('200k_target1.wav');

% Load target 2
[x2, fs2] = audioread('200k_target1.wav');


% Concatenate warm-up and original audio
padded_audio = [x2; x];

% Save to new file
audiowrite('200k_target1_with_warmup.wav', padded_audio, fs);



afr = dsp.AudioFileReader('200k_target1_with_warmup.wav');
fs = afr.SampleRate;

frameSize = ceil(5e-3*fs);
overlapSize = ceil(0.8*frameSize);
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

speechProbLog = []; 

while ~isDone(afr)
    x = afr();
    n = write(inputBuffer,x);

    overlappedInput = read(inputBuffer,frameSize,overlapSize);

    p = VAD(overlappedInput);

    pHold(end) = p;
    scope(x,pHold)

    player(x);

    pHold(:) = p;

    speechProbLog = [speechProbLog; p];
end

t = (0:length(speechProbLog)-1) * hopSize/fs;
plot(t, speechProbLog), xlabel('Time (s)'), ylabel('Speech Probability')