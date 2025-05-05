function slideshow(t)

if ~exist('t', 'var')
	t = 1;
end
h = sort(get(0, 'Children')).';

for ii = h
	figure(ii)
	if t
		pause(t)
	else
		pause
	end
end