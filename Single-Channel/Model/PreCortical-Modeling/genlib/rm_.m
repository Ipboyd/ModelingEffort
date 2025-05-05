function out = rm_(in)
% replaces all _ with \_ (the escape sequence for _ in MATLAB)

if ischar(in)
	in = {in};
	waschar = 1;
else
	waschar = 0;
end

for ele = 1:length(in)
	inds = [1 (find(([in{ele} ' '] == '_').*([' ' in{ele}] ~= '\')))];

	out{ele} = [];
	for i = 1:length(inds) - 1
		out{ele} = [out{ele} in{ele}(inds(i):inds(i + 1) - 1) '\'];
	end

	out{ele} = [out{ele} in{ele}(inds(i + 1):end)];
	
	if numel(inds) == 1
		out{ele} = in{ele};
	end
end

if waschar
	out = out{1};
end