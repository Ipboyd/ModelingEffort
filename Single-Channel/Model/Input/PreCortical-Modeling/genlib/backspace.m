function varargout = backspace(n)

if n > 0
	s = sprintf(repmat(sprintf('\b'), 1, n));
else
	s = [];
end
if nargout == 0
	fprintf(s);
else
	varargout{1} = s;
end