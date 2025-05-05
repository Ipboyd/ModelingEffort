function y = conv(x1, x2, shape)
% function y = conv(a,b)
%   Y = CONV(X1, X2) convolves vectors A and B. The resulting vector is
%   length LENGTH(A)+LENGTH(B)-1. If A and B are vectors of polynomial
%   coefficients, convolving them is equivalent to multiplying the two
%   polynomials.
%
%   It is advantageous to use frequency domain multiplication instead of
%   time-domain convolution when the signal is relatively large. The time-
%   domain method performs NxM multiplications where N and M are the
%   lengths of the two vectors.  FFT-based convolution performs 2 FFT
%   operations at a cost of L*log2(L)/2 where L is the FFT length.  (The
%   FFT length is given by L = pow2(ceil(log2(N+M-1))).) It then performs L
%   pointwise multiplications for a total cost of L*(1+log2(L)).  The cost
%   ratio is therefore L*(1+log2(L))/(N*L), or (1+log2(L))/N which is
%   (1+ceil(log2(N+M-1)))/N. Therefore FFT-based convolution is 
%   advantageous when ceil(log2(N+M-1)) is less than N-1.
%

if nargin < 3
	shape = 'full';
end
na = length(x1);
nb = length(x2);

if na ~= numel(x1) || nb ~= numel(x2)
    error('MATLAB:conv:AorBNotVector', 'A and B must be vectors.');
end

len = na + nb - 1;
nMin = min(na,nb);

if (ceil(log2(len)) < nMin-1) && ((sum(abs(fix(x1) - x1)) + sum(abs(fix(x2) - x2))) > 0)
%     disp('1')
    if size(x1,1) == 1 && size(x2,1) == 1
        y = real(ifft(fft(x1, len).*fft(x2, len)));
    else
        y = real(ifft(fft(x1(:), len).*fft(x2(:), len)));
    end
else
%     disp('2')
    if na > nb
        [y,zf] = filter(x2, 1, x1);
         y(na+1:len) = zf;
    else
        [y,zf] = filter(x1, 1, x2);
        y(nb+1:len) = zf;
    end
end

switch lower(shape)
	case 'full'
	case 'same'
		y = y(floor(nb/2) + (1:na));
	case 'valid'
		y = y(nMin:end-nMin+1);
	otherwise
		error('Shape must be ''full'', ''same'', or ''valid''');
end
