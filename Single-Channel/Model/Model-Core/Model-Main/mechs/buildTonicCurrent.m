function I = buildTonicCurrent(T,N_pop,dt,numLocs)

% tonic current is applied to TD cell at focused location, driving
% disinhibition during stimulus playback

t_on = 250; % [ms]
t_len = 3000; % [ms] 

sim_len = numel(T);

padToTime = sim_len / numLocs;
I = zeros(sim_len,N_pop);

for i = 1:numLocs
    I((t_on/dt+1:t_on/dt+t_len/dt)+padToTime*(i-1),:) = 1;
end

end