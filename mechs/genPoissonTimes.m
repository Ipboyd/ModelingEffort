function token = genPoissonTimes(N_pop,dt,FR,std,locNum)

if ~isempty(locNum)
    len = 35000;
else
    len = 35000*24;
end

temp = (rand(len,N_pop) < (FR + std*randn(len,N_pop))*dt/1000);

refrac = 1.5;  % ms

% delete spikes that violate refractory period
for i = 1:N_pop
    spk_inds = find(temp(:,i));
    ISIs = diff(spk_inds);
    temp(spk_inds(find(ISIs < refrac/dt)+1),i) = 0;
end

token = temp;

end