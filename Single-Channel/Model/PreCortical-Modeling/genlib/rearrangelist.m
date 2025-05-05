function x = rearrangelist(x, n, order, dim)

s = size(x);
ndims = length(s);
if nargin < 3
	error('At least three inputs needed.')
elseif s(1) > 1 || ndims > 2
	dim = 1;
else
	dim = 2;
end
permOrder = [1:dim - 1, dim + 1:ndims, dim];
s = s(permOrder);
x = permute(x, permOrder);
x = reshape(x, [s(1:ndims - 1) n]);
x = permute(x, [1:ndims - 1, order + ndims - 1]);
x = reshape(x, s);
x = ipermute(x, permOrder);