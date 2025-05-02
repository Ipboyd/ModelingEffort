function p = rms(x)

% if isreal(x)
% 	p = sqrt(mean(x.^2));
% else
p = sqrt(mean(abs(x).^2));
% end
