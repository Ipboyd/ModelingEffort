function token = genPoissonTimes(N_pop,dt,FR,std,simlen)

if isempty(simlen)
    simlen = 35000;
end

%Temp changing this to see if it effects network
temp = (rand(simlen,N_pop) < (FR + std*randn(simlen,N_pop))*dt/1000);   

refrac = 1;  % ms

% delete spikes that violate refractory period
for i = 1:N_pop
    spk_inds = find(temp(:,i));
    ISIs = diff(spk_inds)*dt; 
    temp(spk_inds(find(ISIs < refrac)+1),i) = 0;
end

% token = temp*0;
token = temp;

end