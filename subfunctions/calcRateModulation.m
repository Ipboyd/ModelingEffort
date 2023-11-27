function RM = calcRateModulation(spks,t_stim)

% from Downer et al 2021: difference between max and min FR divided by sum

dt = 0.1; % ms

t_vec = ( 0 : 20 : t_stim*1000) + 250; % in ms, accounts for first 250ms of silence
% k = exp(-(0:1:100)/10); % decaying exponential kernel with decay = 10ms

% calculate maximum and minimum FR per trial

maxFR = [];
minFR = [];

for n = 1:size(spks,2)
    spk_times = find(spks(:,n))*dt; % ms
    FR = histcounts(spk_times,t_vec); % no need to normalize

    %FR = conv(FR,k);
    %FR(length(t_vec)+1:end) = [];

    maxFR = cat(1,maxFR,max(FR));
    minFR = cat(1,minFR,min(FR));
end

% average across trials to calculate modulation
RM = (mean(maxFR) - mean(minFR)) / (mean(maxFR) + mean(minFR));

end