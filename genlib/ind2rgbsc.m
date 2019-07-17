function y = ind2rgbsc(x, cm, clim)

if ~exist('clim', 'var')
	clim = [min(x(:)) max(x(:))];
end

n = size(cm, 1);
x = round((min(clim(2), max(clim(1), x)) - clim(1))/diff(clim)*(n - 1)) + 1;
y = zeros([size(x) 3]);

for ii = 3:-1:1
	y(:,:,ii) = reshape(cm(x,ii), size(x));
end
