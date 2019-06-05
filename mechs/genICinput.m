function sigIn = genICinput(trial, tauR, tauD, dt)
% input:
%   tauR, tauD = rise and fall times of the EPSP waveform
%   dt = sampling frequency of IC data
% output:
%   sigIn = a matrix of EPSP waveforms, with dimensions time x (nfreqs x nlocs)
%
% @ Erik Roberts, Kenny Chou
% Boston Univeristy, 2019
%
% TODO: allow resampling of spk_IC to different dt

if exist('IC_spks.mat','file')
    fileData = load('IC_spks.mat','spks');
else
    fileData = load('..\IC_spks.mat','spks');
end

% IC data: trial x location x time
% Desired: time x location x trial
% Permute data 
spk_IC = fileData.spks;
spk_IC = permute(spk_IC,[3,2,1]);
sigIn = squeeze(spk_IC(:,:,trial)); % time x location x cells


% ========================= create EPSP waveform =========================
% ============= convolve each time series with epsc waveform ==============
t_a = max(tauR,tauD)*7; % Max duration of syn conductance
t_vec = 0:dt:t_a;
tau2 = tauR;
tau1 = tauD;
tau_rise = tau1*tau2/(tau1-tau2);
b = ((tau2/tau1)^(tau_rise/tau1) - (tau2/tau1)^(tau_rise/tau2)^-1); % /tau2?
epsc =  - b * ( exp(-t_vec/tau1) - exp(-t_vec/tau2) ); % - to make positive
sigIn = conv2(sigIn,epsc','same');

end