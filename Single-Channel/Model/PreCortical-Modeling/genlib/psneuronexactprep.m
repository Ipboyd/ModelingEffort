function [expTable,logTable,rhoTable,dx] = psneuronexactprep(taue,taum,dx)
if(nargin < 2)
    error('At least two inputs required:\n taue and taum (synaptic and membrane time constants)');
elseif(nargin<3)
    dx = 1e-5;
end
taus=taue/taum;

x = (0:floor(2/dx)-1)*dx; % Inputs 0->2
expTable = exp(-x);
x = (0:floor(2/dx)-1)*dx; % Inputs 0->2
logTable = log(x);
x = (0:floor(2/dx)-1)*dx; % Inputs 0->2 yields 1% out of bounds (->3 is 0.01%)
rhoTable = (rho(x,taus)*taus.*x);

function sum1 = rho(g,taus)

MAX_GAMMA_ERROR = 1e-15;
x = taus*g;
sum1 = ones(size(x))/(1-taus);
for ii = 1:length(x)
    ap = 2 - taus;
    del = 1/(1-taus);

    del = del*x(ii)/ap;
    sum1(ii) = sum1(ii) + del;
    while (del >= sum(ii) * MAX_GAMMA_ERROR)
        ap = ap+1;
        del = del*x(ii)/ap;
        sum1(ii) = sum1(ii)+del;
    end
end

% % This is actually slower!
% maxcount = 16;
% invtaus = 1/(1-taus);
% del = (taus*g)./(2 - taus)*invtaus;
% xoap = repmat(taus*g,maxcount,1)./repmat((2 - taus)+(1:maxcount).',1,size(g,2));
% xoapcumdel = cumprod(xoap,1).*repmat(del,maxcount,1);
% sum1 = sum([ones(size(g))*invtaus + del;xoapcumdel],1);
