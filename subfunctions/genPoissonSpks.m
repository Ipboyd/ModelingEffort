function [S] = genPoissonSpks(spike_rate,T,t_abs,t_rel,rec)

% Inputs:
%   [spike_rate]
%   [T]          time vector for spike rate (unit: ms)
%   [t_abs]      absolute refractory period (unit: ms)
%   [t_rel]      relative refractory period (unit: ms)
%   [rec]        recovery speed

% Outputs:
%   [S]          array with 1's for all spike times

rng('shuffle');

%% Spike generation
dt = T(2)-T(1); % ms
N = length(spike_rate);

n_refab = 6/dt;     % length of refractory period time in samples
tw = 0:n_refab;     % sample time vector for recovery rate

t_abs_samp = t_abs/dt;
t_rel_samp = t_rel/dt;

w = (tw - t_abs_samp).^rec ./ ((tw - t_abs_samp).^rec + (t_rel_samp).^rec); % recovery function (schaette et al 2005)
w(tw < t_abs_samp) = 0;

x = rand(1,N); S = zeros(1,N);
spike_times = [];
for i = 1:N   % sample
    if ~isempty(spike_times) && i-spike_times(end) < n_refab
        spike_rate(i)=spike_rate(i) * w(i-spike_times(end)); % implement relative refractory period
    end
    if x(i)<dt*spike_rate(i)
        S(i) = 1;
        spike_times = [spike_times; i];
    end
end

end
