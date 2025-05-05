function str = sig2str(p,m)
% function str = sig2str(p,m)
%   STR = SIG2STR(P,M) returns a string, e.g. 'p = 0.01', where the 
%   significance value P has been truncated to one significant digit. If P
%   is less than the minimum probability M (optional, default = 0.001),
%   then 'p < 0.001' is returned.
%   

if(nargin < 1)
    error('Too few arguments to sig2str')
elseif(nargin < 2)
    m = 0.001;
end

nsig = -floor(log10(p));
tenfact = 10^nsig;
pmod = ceil(p*tenfact)/tenfact;

if(pmod < m || isinf(tenfact))
    str = sprintf(sprintf(' < %%0.%dg',-floor(log10(m))),m);
else
    str = sprintf(sprintf(' = %%0.%dg',nsig),pmod);
end;
