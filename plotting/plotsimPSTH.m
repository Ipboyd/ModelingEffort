function plotsimPSTH(snn_out,ind,field,ch)

t_vec = 0:200:35000;

psth = histcounts(find(snn_out.(field)(35000*(ind-1)+1:35000*ind,ch)),t_vec);
psth(end+1) = 0;

plot(t_vec,psth);


end