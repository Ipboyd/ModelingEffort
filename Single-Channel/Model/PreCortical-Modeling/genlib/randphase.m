function x = randphase(x, amount)

if nargin < 2
	amount = 1;
end

x = real(ifft(fft(x).*exp(1j*amount*angle(fft(rand(size(x)))))));