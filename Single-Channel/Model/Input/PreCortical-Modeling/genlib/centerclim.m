function centerclim(h)

if nargin == 0
	h = gca;
end

cMax = -Inf;
for hi = 1:numel(h)
	cMax = max(cMax, max(abs(get(h(hi), 'CLim'))));
end
set(h, 'CLim', [-1 1].*cMax);