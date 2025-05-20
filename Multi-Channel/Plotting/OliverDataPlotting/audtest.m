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

VAD = voiceActivityDetector('FFTLength',frameSize);

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
% plot(t, speechProbLog), xlabel('Time (s)'), ylabel('Speech Probability')

count = 0;
flag = 0;
indexer = [];

for k = 1:length(speechProbLog)

    if speechProbLog(k) >= 0.95
        count = count+1;
        
    end

    if count == 35
        indexer = [indexer,0];
        indexer(end - 35 + 1 : end) = 1; %This should replace instead of add
        flag = 1;
        count = 0;
    elseif flag == 1;
        indexer = [indexer,1];
        
        if speechProbLog(k) < 0.95
            flag = 0;
        end
       

    else 
        indexer = [indexer,0];
        
        

    end

end

%% 

idx_nor = 10 .* indexer;
ix = 4.6432e-04:4.6432e-04:3;

op_idx = 10 - idx_nor;
op_idx = abs(op_idx);

figure;
 
%plot(ix,idx_nor)
area(ix,idx_nor,'FaceColor',[0.9882 0.7804 0.7804],'EdgeColor',[0.9882 0.7804 0.7804]); hold on
area(ix,op_idx,'FaceColor',[0.7804 0.8392 0.9882],'EdgeColor',[0.7804 0.8392 0.9882]); hold off
 
ylim([-0.5 10])


