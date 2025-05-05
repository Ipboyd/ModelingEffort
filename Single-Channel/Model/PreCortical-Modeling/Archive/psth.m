function psth(in,t)

t_vec = 0:20:(max(t*1000)+20);
frate = histcounts(in,t_vec);

plot(t_vec(1:end-1),frate); xlim([0 max(t_vec)]);


