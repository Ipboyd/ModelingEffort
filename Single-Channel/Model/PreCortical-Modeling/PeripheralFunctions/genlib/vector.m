function b = vector(a,preserveCellness)

% vec = vector(mat)
%  returns mat(:), but allows you to do it without mat being a variable in your
%  workspace.

if ~iscell(a) || (nargin>1 && preserveCellness)
	b = a(:);
else
	s = zeros(numel(a), 1);
	for ii = 1:numel(a)
		s(ii) = numel(a{ii});
	end
	b = zeros(sum(s), 1);
	pos = 1;
	for ii = 1:numel(a)
		b(pos + (0:s(ii) - 1)) = a{ii}(:);
		pos = pos + s(ii);
	end
end