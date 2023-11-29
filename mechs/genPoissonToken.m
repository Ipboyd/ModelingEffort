function token = genPoissonToken(N_pop,dt,FR,std,tauR,tauD,locNum)

%%%%%%%% calculate post-synaptic current %%%%%%%%%%
t = 0:0.1:300;
tau_rise = tauD*tauR/(tauD-tauR);
b = ((tauR/tauD)^(tau_rise/tauD) - (tauR/tauD)^(tau_rise/tauR))^-1;
f =  b * ( exp(-t/tauD) - exp(-t/tauR) );

if ~isempty(locNum)
    len = 35000;
else
    len = 35000*24;
end

temp = (rand(len,N_pop) < (FR + std*randn(len,N_pop))*dt/1000);

% delete spikes that violate refractory period
for i = 1:N_pop
    spk_inds = find(temp(:,i));
    ISIs = diff(spk_inds);
    temp(spk_inds(find(ISIs < 1.5/dt)+1),i) = 0;
end

token = zeros(len+length(f)-1,N_pop);

% convolve token with EPSC
for i = 1:N_pop
    token(:,i) = fakeConv(f,temp(:,i));
end

% token should have the same number of elements as simulation
token((len+1):end,:) = [];

end

function token = fakeConv(kern,temp)

temp = temp';

token = zeros(1,length(kern)+length(temp)-1);
kern_len = numel(kern);

i = find(temp);

for j = 1:length(i)
    token(i(j):i(j)+kern_len-1) = temp(i(j):i(j)+kern_len-1) + kern;
end

end