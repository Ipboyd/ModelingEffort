function invar = invariance(x, y, t, c, r)
%function invar = invariance(x, y, t, c, r)
% Computes the moment of inertia invariance metric.  Set r to 0 or leave it
% empty to just do the area.
% 
% x is list of x-coordinates
% y is list of percent corrects
% t is template index
% c is chance
% r is the radius power (can be left out, defaults to 0)

if nargin < 5
	r = 0;
end
if max(y) > 1
	y = y/100;
end
if c > 1
	c = c/100;
end

ysize = size(y);
if length(ysize) == 2 && ysize(1) == 1
	y = y(:);
end

x = x(:);
i = 1:length(x) - 1;
d = repmat((x(i + 1) - x(i)).*abs(x(t) - x(i + (i >= t))).^r, 1, size(y(:,:), 2));
invar = 1 - sum(d.*abs(repmat(y(t, :), i(end), 1) - y(i + (i >= t), :)))./((1 - c)*sum(d));

if length(ysize) > 2;
	invar = reshape(invar, ysize(2:end));
end