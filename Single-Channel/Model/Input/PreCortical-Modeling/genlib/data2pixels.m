function [x, y] = data2pixels(varargin)
% takes plotted data and converts them to units of pixels, using the same origin
if nargin == 2
	hAx = gca;
	x = varargin{1};
	y = varargin{2};
elseif nargin == 3
	hAx = gca;
	x = varargin{1};
	y = varargin{2};
end
unitsAx = get(hAx, 'Units');
set(hAx, 'Units', 'pixels');

posAx = get(hAx, 'Position');

xLim = get(hAx, 'XLim');
yLim = get(hAx, 'YLim');

x = x/diff(xLim)*posAx(3);
y = y/diff(yLim)*posAx(4);

set(hAx, 'Units', unitsAx)
