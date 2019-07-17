function in = rmtok(in, token)

if ischar(in)
	in = {in};
	waschar = 1;
else
	waschar = 0;
end

for ele = 1:length(in)
	slen = length(in{ele});
	tlen = length(token);

	if tlen > slen
		return;
	end

	inds = [];
	for ii = 1:slen - tlen + 1
		if strcmp(token, in{ele}(ii:ii + tlen - 1))
			inds = ii:ii + tlen - 1;
		end
	end

	in{ele}(inds) = [];
end

if waschar
	in = in{1};
end