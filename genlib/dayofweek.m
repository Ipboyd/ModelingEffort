function s = dayofweek(t)

if nargin == 0
	t = clock;
end
d = 'fsumtwr';
n = mod(floor(datenum(t)), 7) + 1;
s = d(n);