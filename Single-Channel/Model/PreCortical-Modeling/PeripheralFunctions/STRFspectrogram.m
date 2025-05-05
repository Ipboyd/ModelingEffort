function [stim_spec,t,f]=STRFspectrogram(stim,fs)
%% Inputs
%   stim: array or cell of sound pressure waveform
%   fs
% Outputs
%   stim_spec: 
%       stim_spec.w1 STRF matrix
%       stim_spec.t  time array for STRF
%       stim_spec.f  freq array for STRF

preprocStimParams = struct;      %create preprocessing param structure
preprocStimParams.tfType = 'stft'; %use short-time FT
preprocStimParams.rawSampleRate=fs;
tfParams = struct;               %create time-frequency params
tfParams.high_freq = 8000;       %specify max freq to analyze
tfParams.low_freq = 500;         %specify min freq to analyze
tfParams.log = 1;                %take log of spectrogram
tfParams.dbnoise = 80;           %cutoff in dB for log spectrogram, ignore anything below this
tfParams.refpow = 0;             %reference power for log spectrogram, set to zero for max of spectrograms across stimuli
preprocStimParams.tfParams = tfParams;
% make a temporary directory to store preprocessed sound files (should be
%  specific to parameters for preprocSound)
% tempPreprocDir = tempname();
%tempPreprocDir= [pwd filesep 'resampled-stimuli\temp'];

if ~isfolder(fullfile(pwd, 'resampled-stimuli', 'temp')), mkdir(fullfile(pwd, 'resampled-stimuli', 'temp')); end

%mkdir(tempPreprocDir);
preprocStimParams.outputDir = fullfile(pwd, 'resampled-stimuli', 'temp');
%% use preprocSound to generate spectrogram
% input stim  has to be cell array
if isnumeric(stim) % if stim is a numeric array
    temp=stim;
    stim=cell(1);stim{1}=temp; % convert stim into a cell with same array inside
end
[stim_spec, groupIndex, stimInfo, preprocStimParams] = preprocSound(stim, preprocStimParams);
%% generate corresponding timeline for spectrogram
tInc = 1 / stimInfo.sampleRate;
t = 0:tInc:(size(stim_spec, 1)-1)*tInc;
f=stimInfo.f;