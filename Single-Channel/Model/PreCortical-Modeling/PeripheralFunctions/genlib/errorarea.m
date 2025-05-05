function varargout = errorarea(varargin)
% function varargout = errorarea(x,y,e,...)
%   varargout = errorarea(x,y,e,...) takes vectors x, y, and e as x 
%   locations, y values, and error values e and plots an area() from y+e to
%   y-e at corresponding values of x. The function passes additional
%   arguments ... to the function area(), and returns the handle returned
%   by area();
%
%   This function is useful for plotting standard errors. For example:
%
%   x = 1:20;
%   y = randn(25,20);
%   e = std(y,[],1)/sqrt(25);
%   y = mean(y,1);
%   errorarea(x,y,e,'EdgeColor','none','FaceColor',[0.7 0.7 0.7]); hold on;
%   plot(x,y,'k'); hold off;
%

x = varargin{1};
y = varargin{2};
e = varargin{3};
if nargin >= 4 && isnumeric(varargin{4}) && isequal(size(varargin{4}), [1 3])
	c = varargin{4};
	propsInd = 5;
else
	c = 0.5*[1 1 1];
	propsInd = 4;
end

if(size(x,1) == 1); x = x.'; end
if(size(y,1) == 1); y = y.'; end
if(size(e,1) == 1); e = e.'; end

xx = [x; flipud(x)];
yy = [y+e; flipud(y-e)];

h = patch(xx, yy, c, 'LineStyle', 'None'); % plot with defaults
if nargin >= propsInd
	set(h, varargin{propsInd:end}) % if stuff is specified, change it to match those specs
end
if nargout == 1
	varargout(1) = {h};
end