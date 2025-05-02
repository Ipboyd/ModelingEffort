function out = cosWindow(sig,stwind,edwind,pow)

if(nargin < 3)
    error('Not enough input arguments')
end
if(nargin < 4)
    pow = 2;
end

if(size(sig,1) == 1)
    out = transpose(sig);
else
    out = sig;
end

% Rear part
slen = size(out,1);
si = max(1,slen-edwind+1);
out(si:slen,:) = repmat(cos(pi*(1+max(0,edwind-slen):edwind)/edwind/2)'.^pow, 1, size(out,2)) .* out(si:slen,:);

% Front part
ei = min(slen,stwind);
out(1:ei,:) = repmat(cos(pi*(stwind:-1:1+max(0,stwind-slen))/stwind/2)'.^pow, 1, size(out,2)) .* out(1:ei,:);

if(size(sig,1) == 1)
    out = transpose(out);
end
