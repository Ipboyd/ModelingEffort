function [spike_times, spkcnt] = PSTH2Spks(PSTH,numTrials,gain,dt,noiseMultiplier)
addpath('ICSimStim');
% mean_rate = 100; %100 hz
% PSTH = PSTH/mean(PSTH) * mean_rate;
PSTH = PSTH*gain;
spike_times=cell(numTrials,1);
spkcnt=0;
noiseMultiple = noiseMultiplier;
for i = 1:numTrials
    % convert dt from ms to s
    [~,temptspk]=spike_generator_kc(PSTH,[0 dt],noiseMultiple);
    spike_times{i,1}=temptspk*1000;
    spkcnt=spkcnt+length(temptspk);
end
spkcnt=spkcnt/numTrials;