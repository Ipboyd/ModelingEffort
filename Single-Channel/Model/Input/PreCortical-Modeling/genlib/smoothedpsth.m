function out = smoothedpsth(spikeMat, binTime, fs, method)
% out = smoothpsth(spikeMat, binTime [s], fs [Hz], method [OPTIONAL])
% SPIKEMAT has dimensions [time, trials, stims].
% FS defaults to 1 if not specified or empty.
% BINTIME is the time parameter of the given smoothing method.  See below.
% METHOD can be:
%    'moving' - moving average, BINTIME is kernel length (default, centered)
%    'normal' - gaussian, BINTIME is STD (centered)
%    'exp' - exponential, BINTIME is tau (not centered)
%    any of the methods from the matlab function smooth (centered)

if nargin <= 2
	warning('FS not defined. Defaulting to 1.')
	fs = 1;
end
if nargin == 3
	if isempty(fs)
		fs = 1;
	end
	method = 'moving';
end

binLen = max(1, round(binTime*fs));
meanRates = squeeze(mean(spikeMat, 2))*fs;
s = size(meanRates);
out = zeros(s(1), prod(s(2:end)));

method = lower(method);
switch method
	case 'moving'
		len = s(1) + binLen - 1;
		out = real(ifft(fft(meanRates, len).*repmat(fft(repmat(1/binLen, binLen, 1), len), [1 s(2:end)])));
		out = out(ceil(binLen/2) + 1:end - floor(binLen/2) + 1,:);
	case 'normal'
		out = normconv(meanRates, binLen);		
	case 'exp'
		out = expconv(meanRates, binTime, fs)/fs/binTime;
	case 'binned'
		error('Binned version coming eventually. No promises on when, though.')
	otherwise
		for ii = 1:size(out, 2)
			out(:,ii) = smooth(meanRates(:,ii), binLen, method);
		end
end
out = reshape(out, s);
out(out < 0) = 0;
