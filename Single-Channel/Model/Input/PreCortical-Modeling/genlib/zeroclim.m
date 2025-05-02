function zeroclim(h)

if nargin == 0
	h = gca;
end

set(h, 'CLim', [0 1].*get(h, 'CLim'));