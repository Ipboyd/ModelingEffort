function x = hermit(x)
% Forces Hermitian symmetry on input. Works along the first dimension,
% unless input is a row vector.

s = size(x);
if s(1) == 1 && length(s) == 2 % check if it's a row vector
	x = x.';
	s(1) = s(2);
	row = true;
else
	row = false;
end

x(end - floor(s(1)/2 - 1):end,:) = conj(x(floor(s(1)/2) + 1:-1:2,:));

if row
	x = x.';
end