% original stimuli are in 195312 Hz, and STRFspectrogram likes sampling
% rates that are multiples of stimSampleRate = 10000

% original 195312 Hz stimuli should be in archive/fixed-stimuli (don't use
% stimuli, which doesn't have the first 250 ms of zero padding)

mkdir(fullfile('..','resampled-stimuli'));

[y1,fs]=audioread('200k_target1.wav');
y2=audioread('200k_target2.wav');

y1_new = resample(y1,200000,fs);
y2_new = resample(y2,200000,fs);

for m = 1:10
    msig = audioread(sprintf('200k_masker%i.wav',m));
    m_new = resample(msig,200000,fs);
    audiowrite(fullfile('..','resampled-stimuli',sprintf('200k_masker%i.wav',m)),m_new,200000);
end
audiowrite(fullfile('..','resampled-stimuli','200k_target1.wav'),y1_new,200000);
audiowrite(fullfile('..','resampled-stimuli','200k_target2.wav'),y2_new,200000);

length(y1_new)/200000