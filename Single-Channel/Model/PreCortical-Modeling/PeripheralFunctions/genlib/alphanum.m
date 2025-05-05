function out = alphanum(in)

if ischar(in)
	out = sum(double(in - 'a' + 1).*26.^(length(in) - 1:-1:0));
elseif isfloat(in) && in > 0
	out = [];
	out = [mod(in - 1, 26) + 1, out];
	in = in/26;
	while(ceil(in) > 1)
		out = [mod(in - 1, 26) + 1, out];
		in = in/26;
	end
	out = char(out + 'a' - 1);
else
	error('Invalid input.')
end