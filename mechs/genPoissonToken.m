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

temp = (rand(len,N_pop) < dt/1000*(FR + std*randn(len,N_pop)));

% delete spikes that violate refractory period
for i = 1:N_pop
    spk_inds = find(temp(:,i)); ISIs = diff(spk_inds);
    temp(spk_inds(find(ISIs < 3/dt)+1),i) = 0;
end

% convolve token with EPSC
for i = 1:N_pop
    token(:,i) = conv(f,temp(:,i));
end
token(1+len:end,:) = [];

end