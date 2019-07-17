function out = npickk(n,k)
% function out = npickk(n,k)
%   OUT = NPICKK(N,K) randomly chooses K integers in 1:N.
% 

if(k > n)
    warning('k > n, taking k = n')
    k = n;
end

out = randperm(n);
out = out(1:k);
