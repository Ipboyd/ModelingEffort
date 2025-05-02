function x = stbound(x, bound, noShift, extraShift)
% y = stbound(x, bound)
% Limit the spike times contained in x to the interval
% [bound(1), bound(2)] and subtracts bound(1) from all spike times.

if(nargin < 3 || isempty(noShift))
    noShift = false;
end
if(nargin < 4 || isempty(extraShift))
    extraShift = 0;
end

s = size(bound);
[nTrials,nStim,nOther] = size(x);

if(s(1) == 1) % Row vector, should be 2 wide
    if(s(2) == 2)
        bound = repmat(bound,nStim,1);
    else
        error('wrong inputs')
    end
elseif(s(2) == 1) % Column vector, should be 2 tall
    if(s(1) == 2)
        bound = repmat(bound.',nStim,1);
    else
        error('wrong inputs')
    end
elseif(s(2) ~=2 || s(1) ~= nStim)
    error('wrong inputs');
end

leftShifts = bound(:,1);
if(noShift)
    leftShifts(:) = 0;
end
if(extraShift)
    leftShifts(:) = leftShifts(:) - extraShift;
end
    
for oi = 1:nOther
    for si = 1:nStim
        for ii = 1:nTrials
        	x{ii,si,oi} = x{ii,si,oi}(logical((x{ii,si,oi} >= bound(si,1)) .* (x{ii,si,oi} <= bound(si,2)))) - leftShifts(si);
        end
    end
end
