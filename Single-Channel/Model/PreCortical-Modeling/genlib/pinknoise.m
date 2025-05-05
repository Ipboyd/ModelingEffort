function out = pinknoise(Ns,Ncol)
%PINKNOISE Pink noise generation.
%   OUT = PINKNOISE(NS,NCOL) returns NCOL columns of pink noise of length
%   NS samples each. NCOL and NS must be real scalars. Note that due to the
%   fractal nature of pink noise, supplying FS is not necessary.
%
%   Code simplified from original:
%   http://www.mathworks.com/support/solutions/archived/1-16842.html
%
%   Values generally stay between +/- 1, but there's no guarantee here.
%
%   Written July 17 2007
%   Eric Larson (edlarson@bu.edu)

a = ones(1,200);
for ii = 2:size(a,2)
    a(ii) = (ii - 2.5) * a(ii-1) / (ii-1);
end

out = filter(1,a, 0.05 * randn(Ns,Ncol));