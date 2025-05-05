function varargout = fillarea(varargin)

x = varargin{1};
y1 = varargin{2};
y2 = varargin{3};

if(size(x,1) == 1); x = x.'; end
if(size(y1,1) == 1); y1 = y1.'; end
if(size(y2,1) == 1); y2 = y2.'; end

varargin{1} = [x; flipud(x)];
varargin{2} = [y2; flipud(y1)];
varargin = varargin([1 2 4:nargin]);

h = area(varargin{:});
if nargout == 1
	varargout(1) = {h};
end