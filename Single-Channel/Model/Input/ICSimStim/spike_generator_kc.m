function [spike_train,spike_times]=spike_generator_kc(spike_rate,time,noiseMultiple)
% Inputs:
%   [spike_rate]
%   [time]          time vector for spike rate (unit: s)
% Outputs:
%   [spike_train]        array with 1's for all spike times
%   [spike_times]        unit in s

rng('shuffle');

%% Spike generation
dt=time(2)-time(1); %s
n=length(spike_rate);
tw=0:dt:6/1000; % (ms) time vector for recovery rate
w=1-1./(1+exp(4000*(tw-3.8/1000)));
n_refab=6/1000/dt; % absolute refractory period

x=rand(1,n)*noiseMultiple;
spike_train=zeros(1,n);
spike_times=[];
for i=1:n
    if ~isempty(spike_times)&&i-spike_times(end)<n_refab
        spike_rate(i)=spike_rate(i)*w(i-spike_times(end));
    end
    if x(i)<dt*spike_rate(i)
        spike_train(i)=1;
        spike_times=[spike_times; i];
    end
end
spike_times=spike_times*dt;
