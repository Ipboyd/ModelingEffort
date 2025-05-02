function signalplot(x, fs, varargin)

if nargin == 1 || isempty(fs)
	fs = 1;
end

if isrow(x)
	x = x.';
end
len = size(x, 1);
t = 0:1/fs:(len - 1)/fs;
plot(t, x, varargin{:})