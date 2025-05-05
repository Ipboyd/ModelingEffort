function fftplot(x, fs, varargin)

if nargin == 1 || isempty(fs)
	fs = 1;
end

if isrow(x)
	x = x.';
end
len = size(x, 1);
f = (0:len - 1)/len*fs;
plot(f, 20*log10(abs(fft(x))), varargin{:})