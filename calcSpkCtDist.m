function dist_mat = calcSpkCtDist(spks,t1,t2)

% spks: cell vector with each cell containing a spike train 

nT = numel(spks);

dist_mat = zeros(nT,nT);

for n1 = 1:nT
    for n2 = n1:nT
        dist_mat(n1,n2) = abs(numel(spks{n1}(spks{n1} >= t1 & spks{n1} < t2))-numel(spks{n2}(spks{n2} >= t1 & spks{n2} < t2)));
        dist_mat(n2,n1) = dist_mat(n1,n2);
    end
end

end