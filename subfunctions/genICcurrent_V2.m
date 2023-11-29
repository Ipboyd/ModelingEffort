function sigIn = genICcurrent_V2(trial,locNum,label,g_IC,noise)
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

if exist(['IC_current_' label(2:end-1) '.mat'],'file')    % need the (2:end-1) cuz the ' ' count as characters
    fileData = load(['IC_current_' label(2:end-1) '.mat'],'spks');
else
    fileData = load(['..' filesep 'IC_current_' label(2:end-1) '.mat'],'spks');
end

% IC data: trial x location x time
% Desired: time x location x trial
% Permute data 
spk_IC = fileData.spks;
spk_IC = permute(spk_IC,[3,2,1]);
sigIn = -g_IC * squeeze(spk_IC(:,:,trial)) / 1000; % time x location x cells

N_pop = size(sigIn,2);

if ~isempty(locNum)
   sigIn = sigIn(35000*(locNum-1)+1:35000*locNum,:);
   
   % zero mean
   sigIn = sigIn - mean(sigIn);
   
   scFactor = std(sigIn);
   
   % unity variance
   sigIn = sigIn/scFactor;    % this changes per location
   
   sigIn(2501:32500,:) = sigIn(2501:32500,:) + noise*randn(30000,N_pop);
else
    for t = 1:24
        sigIn(2501+(35000*(t-1)):32500+(35000*(t-1)),:) = ...
            sigIn(2501+(35000*(t-1)):32500+(35000*(t-1)),:) + noise*randn(30000,N_pop);
    end
end

end