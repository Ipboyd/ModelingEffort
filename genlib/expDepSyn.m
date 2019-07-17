function depVec = expDepSyn(x,taud,fs,p)
%function depVec = expDepSyn(x,taud,fs,p)
% Model for synaptic depression/facilitation developed by Kamal et al.

if(nargin ~= 4)
    error('Four (4) inputs required')
elseif(~isequal(size(taud),size(p)) || ~isvector(taud) || ~isvector(p))
    error('taud and p need to be vectors of the same length');
elseif(size(x,1) == 1)
    x = x.';
end
if(~isa(x,'double'))
    x = double(x);
end

depVec = ones(size(x));
for ii = 1:length(taud)
    if(p(ii) < 1) % Multiplicative scheme
        depVec = depVec .* expmult(x,taud(ii),fs,p(ii));
    elseif(p(ii) > 1)
        depVec = depVec .* (1 + (p(ii)-1)*([zeros(1,size(depVec,2));expconv(x(1:end-1,:),taud(ii),fs)] + max(x-1,0)/2));
    end
end
