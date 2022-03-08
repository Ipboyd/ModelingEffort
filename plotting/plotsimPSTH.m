function plotsimPSTH(pop,snn_out)

jump = length(snn_out)/20;

trials = 1:jump:jump*10;

t_vec = 0:20:3500; % ms

temp = find([snn_out(trials).([pop '_V_spikes'])]);
[inds,~] = ind2sub([35000 10],temp);

psth = histcounts(inds/10,t_vec);
psth(end+1) = 0;

plot(t_vec,psth,'linewidth',1); hold on;
xlabel('Time (ms)');
end