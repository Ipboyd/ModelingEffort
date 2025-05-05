function x = cosenv(t, n, fs)
% x = cosenv(t, n, fs)
%    t   duration of onset and offset envelopes
%    n   length of entire envelope in samples (or size of matrix).
%    fs  sample rate

n = n(:).';
if numel(n) == 1
	n(2) = 1;
end

if n == 1;
	x = 1;
else
	x = ones(n(1), 1);
	len = floor(t*fs);
	if n(1) < len
		error('The ramp time is greater than the length of the envelope.')
	elseif n(1) < 2*len
		warning('The onset and offset ramps will overlap.')
	end
	
	x(1:len) = x(1:len).*sin((1:len)/(len + 1)*pi/2).^2.';
	x(n(1) - len + 1:n(1)) = x(n(1) - len + 1:n(1)).*x(len:-1:1);
	
	x = repmat(x, [1 n(2:end)]);
end