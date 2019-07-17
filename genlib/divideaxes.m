function h = divideaxes(varargin)

if nargin == 0
	error('DIVIDEAXES requires at least one input.')
elseif all(strcmpi(get(varargin{1}, 'Type'), 'axes'))
	ha = varargin{1};
	grid = varargin{2};
	if nargin == 3
		space = varargin{3};
	else
		space = [0 0];
	end
elseif numel(varargin{1}) == 2
	grid = varargin{1};
	ha = gca;
	if nargin == 2
		space = varargin{2};
	else
		space = [0 0];
	end
else
	error('Arguments not understood')
end
if numel(grid) == 1
	grid = [grid, grid];
end

if numel(space) == 1
	space = [space space];
end

nAx = numel(ha);
h = zeros([grid, size(ha)]);
for axNum = 1:nAx
	axPos = get(ha(axNum), 'Position');
	axUnits = get(ha(axNum), 'Units');
	
	ratio = 1./(grid([2 1]) - space([2 1]));
	newAxInc = axPos([3 4]).*ratio;
	newAxDims = axPos([3 4]).*(1 - space([2 1])).*ratio;
	
	for rowNum = grid(1):-1:1
		for colNum = 1:grid(2)
			h(rowNum,colNum,axNum) = axes('Units', axUnits, 'Position', [axPos([1 2]) + newAxInc.*[colNum - 1, rowNum - 1], newAxDims]);
		end
	end
end
h(:) = h(end:-1:1,:);
h = permute(h, [3:ndims(h), 1:2]);
delete(ha)
