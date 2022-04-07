function th = genRandomThresholds(N_pop,n,avg,locNum)

if ~isempty(locNum)
    th = gaminv(rand(35000,N_pop),n,avg);
else
    th = gaminv(rand(35000*24,N_pop),n,avg);
end

end