function I = buildTonicCurrent(T,N_pop)

% tonic current is applied to TD cell at focused location, driving
% disinhibition

t_on = 2500;
t_len = 30000;

sim_len = numel(T);
I = zeros(sim_len,N_pop);

if sim_len == 35000
    I(t_on+1:t_on+t_len,:) = 1;
else
    for z = 1:24
        I((t_on+1:t_on+t_len)+35000*(z-1),:) = 1;
    end
end

end