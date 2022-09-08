function s = genPoissonInputs(trial,locNum,label,t_ref,t_ref_rel,rec)

dt = 0.1;  % ms
type = label(2:end-1);   % dynasim reads the apostrophes as literals
fileData = load(['IC_spks_' type '.mat'],'spks');

temp = fileData.spks;
trial_rate = squeeze(temp(:,:,trial)); % time x location x cells

if ~isempty(locNum)
    rate = trial_rate(35000*(locNum-1)+1:35000*locNum,:);
else
    rate = trial_rate; 
end

s = zeros(size(rate));

for n = 1:size(rate,2)
    s(:,n) = spikeGenerator(rate(:,n),dt,t_ref,t_ref_rel,rec);
end

end

function spike_train = spikeGenerator(rate,dt,t_ref,t_ref_rel,rec)

dt_sec = dt/1000; % from ms to s
n = length(rate);

n_refab = 15/1000/dt_sec;  % length of refractory period function (samples)
tw = 0:n_refab; % sample time vector for recovery rate

t_ref_samp = t_ref/1000/dt_sec;
t_rel_samp = t_ref_rel/1000/dt_sec;

w = (tw - t_ref_samp).^rec ./ ((tw - t_ref_samp).^rec + (t_rel_samp).^rec); % recovery function (schaette et al 2005)
w(tw < t_ref_samp) = 0;

x=rand(1,n);
spike_train=zeros(1,n);
spike_times=[];
for i=1:n   % sample
    if ~isempty(spike_times) && i-spike_times(end) < n_refab
        rate(i)=rate(i)*w(i-spike_times(end)); % implement relative refractory period
    end
    if x(i)<dt_sec*rate(i)
        spike_train(i) = 1;
        spike_times=[spike_times; i];
    end
end

end