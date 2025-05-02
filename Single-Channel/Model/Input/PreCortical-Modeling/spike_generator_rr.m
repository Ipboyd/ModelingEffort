function [spike_train,spike_times]=spike_generator_rr(spike_rate,time,t_ref,t_ref_rel,rec)
% Inputs:
%   [spike_rate]
%   [time]          time vector for spike rate (unit: s)
%   [t_ref]         refractory period (unit: ms)
% Outputs:
%   [spike_train]        array with 1's for all spike times
%   [spike_times]        unit in s

rng('shuffle');

%% Spike generation
dt=time(2)-time(1); %s
n=length(spike_rate);

n_refab=6/1000/dt;  % length of refractory period time
tw = 0:n_refab; % sample time vector for recovery rate

t_ref_samp = t_ref/1000/dt;
t_rel_samp = t_ref_rel/1000/dt;

% rec = 2.5;
w = (tw - t_ref_samp).^rec ./ ((tw - t_ref_samp).^rec + (t_rel_samp).^rec); % recovery function (schaette et al 2005)
w(tw < t_ref_samp) = 0;

x=rand(1,n);
spike_train=zeros(1,n);
spike_times=[];
for i=1:n   % sample
    if ~isempty(spike_times) && i-spike_times(end) < n_refab
        spike_rate(i)=spike_rate(i)*w(i-spike_times(end)); % implement relative refractory period
    end
    if x(i)<dt*spike_rate(i)
        spike_train(i)=1;
        spike_times=[spike_times; i];
    end
end
spike_times=spike_times*dt; % convert samples to s
