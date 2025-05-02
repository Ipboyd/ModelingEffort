function varargout = alphapoints(x,y,radius,color,alpha,Ncircle)
% H = ALPHAPOINTS(X,Y,COLOR,WIDTH,ALPHA,NCIRCLE)
%   This function plots "points" that are squares of radius WIDTH, color 
%   COLOR, and (optional) alpha ALPHA (scalars) at the specified paired 
%   points in vectors or matrices X and Y. The parameter NCIRCLE (optional)
%   defines the number of points used to create the circle.

Npts = numel(x);
if(Npts ~= numel(y))
    error('X and Y must be the same size');
end
if(nargin < 5 || isempty(alpha))
    alpha = 1;
end
if(nargin < 6 || isempty(Ncircle))
    Ncircle = 10;
end

% if(isscalar(radius))
%     radius = radius*ones(Npts,1);
% end
% if(isscalar(radius))
%     radius = radius*ones(Npts,1);
% end


t = linspace(0,2*3.1415,Ncircle+1);
t = t(1:Ncircle);
dx = radius*cos(t);
dy = radius*sin(t);
        
oldhold = get(gca,'NextPlot');
if(strcmp(oldhold,'replace')); cla; end
h = zeros(Npts,1);
for pi = 1:Npts
    h(pi) = patch(x(pi)+dx,y(pi)+dy,color,'LineStyle','none','FaceAlpha','flat','FaceVertexAlphaData',alpha); hold on;
end

set(gca,'NextPlot',oldhold);

if nargout == 1
	varargout{1} = h;
end
