function varargout = distmatsupergrid(varargin)

if all(ishandle(varargin{1})) && all(strcmpi(get(varargin{1}, 'type'), 'axes'))
	hDistMatAxes = varargin{1};
	numTargets = varargin{2};
	argNum = 3;
else
	hDistMatAxes = gca;
	numTargets = varargin{1};
	argNum = 2;
end

numAxes = length(hDistMatAxes);
h = zeros(numAxes, 1);
for axNum = 1:numAxes
	xLim = get(hDistMatAxes(axNum), 'XLim');
	yLim = get(hDistMatAxes(axNum), 'YLim');
	
	x = linspace(xLim(1), xLim(2), numTargets + 1);
	y = linspace(yLim(1), yLim(2), numTargets + 1);
	x = x(2:end - 1);
	y = y(2:end - 1);
	
	xx = nan(6*length(x), 1);
	yy = nan(6*length(x), 1);

	xx(1:3:end/2) = x;
	xx(2:3:end/2) = x;
	xx(end/2 + 1:3:end) = xLim(1);
	xx(end/2 + 2:3:end) = xLim(2);
	
	yy(1:3:end/2) = yLim(1);
	yy(2:3:end/2) = yLim(2);
	yy(end/2 + 1:3:end) = y;
	yy(end/2 + 2:3:end) = y;
	
	h(axNum) = line(xx, yy, 'Color', get(hDistMatAxes(axNum), 'XColor'), 'LineWidth', get(hDistMatAxes(axNum), 'LineWidth'));
end

if nargin > argNum
	set(h, varargin{argNum:end})
end

if nargout == 1
	varargout{1} = h;
end