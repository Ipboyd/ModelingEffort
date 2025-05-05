function c = fftconv(a,b,len)
% function out = fftconv(a,b)
%   C = FFTCONV(A, B) convolves vectors A and B using DFTs.
%   The resulting vector is length LENGTH(A)+LENGTH(B)-1.
%   If A and B are vectors of polynomial coefficients, convolving
%   them is equivalent to multiplying the two polynomials.
%
%   It is advantageous to use FFTCONV instead of CONV when the signal is
%   relatively large.  CONV performs NxM multiplications where N and M are
%   the lengths of the two vectors.  FFTCONV performs 2 FFT operations at
%   the cost of L*log2(L)/2 where L is the FFT length.  (The FFT length is 
%   given by L = pow2(ceil(log2(N+M))).) It then performs L pointwise 
%   multiplications for a total cost of L*(1+log2(L)) multiplications.  The
%   cost ratio is therefore L*(1+log2(L))/(N*L) => (1+log2(L))/N which is 
%   approximately log2(L)/N, or ceil(log2(N+M))/N. Therefore FFTCONV is
%   advantageous when ceil(log2(N+M)) is less than N.

if ~isvector(a) || ~isvector(b)
  error('MATLAB:conv:AorBNotVector', 'A and B must be vectors.');
end

len = length(a(:))+length(b(:))-1;
c = ifft(fft(a(:),len).*fft(b(:),len));
