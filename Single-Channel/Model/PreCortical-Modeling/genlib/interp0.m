function y = interp0(x, n)
%INTERP1 1-D interpolation using zero-order-hold
%   XI = INTERP1(X,N) interpolates to find XI, the interpolated/decimated
%   version of X that is N long. It works along the first dimension unless
%   X is a row vector.

error(nargchk(2, 2, nargin));
if isempty(x) % Handle an empty vector input
	y = [];
	warning('MATLAB:emptyVector','Empty vectors can not be interpolated.')
else
	s = size(x);
	if s(1) == 1 && length(s) == 2 % Check if it's a row vector.
		x = x.';
		s(1) = s(2);
		row = true;
	else
		row = false;
	end
	
	y(1:n,:) = x(floor((0:n - 1)/n*s(1)) + 1,:); % Do the interpolation.
	
	if row
		y = y.'; % Make it a row vector if the input was a row vector.
	end
end