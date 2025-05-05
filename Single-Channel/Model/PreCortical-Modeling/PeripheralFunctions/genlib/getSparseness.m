function sparseness = getSparseness(inMat,bs)
% function sparseness = getSparseness(inMat,bs)
%   SPARSENESS = GETSPARSENESS(INMAT,BS) takes a column matrix INMAT 
%   of trial spike trains, computes a PSTH using bin size BS (in 
%   samples), and then computes the sparseness of the responses. 
%   Sparseness values are between 0 and 1.
%
%   This function returns -1 if the spike matrix has no spikes, -2 
%   if the calculated number of bins is <= 1 (bs too large or too 
%   little data), and -3 for both problems
%

[len Ntrials] = size(inMat);
Nbins = ceil(len/bs);

r = histc(mod(find(inMat)-1,len)+1, 0:bs:bs*ceil((len-1)/bs)) / Ntrials / bs;
normfact = sum(r.^2);
if(normfact && Nbins > 1)
    sparseness = (Nbins - sum(r)^2/normfact) / (Nbins - 1);
else
    sparseness = -(~normfact + 2*(Nbins<=1));
end
