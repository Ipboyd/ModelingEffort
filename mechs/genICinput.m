function sigIn = genICinput(trial, ICdir,locNum,label)
% input:
%   tauR, tauD = rise and fall times of the EPSP waveform
%   dt = sampling frequency of IC data
% output:
%   sigIn = a matrix of EPSP waveforms, with dimensions time x (nfreqs x nlocs)
%
% @ Erik Roberts, Kenny Chou
% Boston Univeristy, 2019
%
% 2020-11-17 - removed EPSP waveform estimation
% 2021-01-05 - added EPSP waveform as a square pulse

% ICdir = ICdir(2:end-1); %remove extra ' character from Dynasim Parsing
% label = label(2:end-1);
% if strcmp(label,'I'), locNum = locNum + 1; end
% 
% ICfiles = dir([ICdir filesep '*.mat']);
% % subz = find(~contains({ICfiles.name},'s0') & contains({ICfiles.name},['_' label]));
% targetFileName = [ICfiles(locNum).folder filesep ICfiles(locNum).name];
% fileData = load(targetFileName,'spks');

if exist(['IC_spks_' label(2:end-1) '.mat'],'file')
    fileData = load(['IC_spks_' label(2:end-1) '.mat'],'spks');
else
    fileData = load(['..\IC_spks_' label(2:end-1) '.mat'],'spks');
end


% IC data: trial x location x time
% Desired: time x location x trial
% Permute data 
spk_IC = fileData.spks;
spk_IC = permute(spk_IC,[3,2,1]);
sigIn = squeeze(spk_IC(:,:,trial)); % time x location x cells

% convolve with short square pulse
dt = 0.1; %ms
dur = 0.5; %ms
epsc = ones(1, round(dur/dt));
sigIn = conv2(sigIn,epsc');
sigIn(size(spk_IC,1)+1:end,:) = []; %trim off extras
end