function varargout = arrow(x,y,tiplen,backlen,stwidth,edwidth,headwidth,varargin)
% function varargout = arrow(X,Y,tiplen,backlen,width,headwidth,...)
%   varargout = arrow(X,Y,TIPLEN,BACKLEN,STWIDTH,EDWIDTH,HEADWIDTH,...) draws an 
%   arrow to/from the points specified by the 2-element vectors X and Y. 
%   The arrow has a tip length TIPLEN, backward tip length BACKLEN, width 
%   WIDTH, and headwidth HEADWIDTH as depicted below:
%
%   
%                                      backlen  tiplen
%                                       |--|-----------| 
%                                                        _ _
%                                        \\.              |
%                                         \ '\.           |
%                     _ _              ____\   '\.        |
%         _ _          |   ____,,---'''           '\.     |
% stwidth _|_  edwidth |  |____                      :>   | headwidth
%                     _|_      ''---...____       ./'     |
%                                          /   ./'        |
%                                         / ./'           |
%                                        //'             _|_
%
%
%
%   NOTE: This arrow will only look correct on equal axes. For example, the
%   following code draws an arrow from (0.1,0.2)->(1,0.8):
%     
%     arrow([0.1 1],[0.2 0.8],0.2,0.05,0.01,0.05,0.25);
%     axis([0 1 0 1]);
%     axis equal;
%   

if(numel(x) ~= 2 || numel(y) ~=2)
    error('X and Y must be 2-element vectors');
end
[th,r] = cart2pol(x(2)-x(1), y(2)-y(1));
rotmat = [cos(th) -sin(th); sin(th) cos(th)];

if(nargin < 3 || isempty(tiplen))
    tiplen = 0.25*r;
end
if(nargin < 4 || isempty(backlen))
    backlen = 0;
end
if(nargin < 5 || isempty(stwidth))
    stwidth = 0.05*r;
end
if(nargin < 6 || isempty(edwidth))
    edwidth = stwidth;
end
if(nargin < 7 || isempty(headwidth))
    headwidth = 0.15*r;
end

xx = [0 r-tiplen r-tiplen-backlen r];
xx = [xx fliplr(xx(1:end-1))];

yy = [stwidth/2 edwidth/2 headwidth/2 0];
yy = [yy -fliplr(yy(1:end-1))];

temp = rotmat*[xx;yy];
xx = temp(1,:)+x(1);
yy = temp(2,:)+y(1);

newplot;
h = patch(xx, yy, 'k', 'LineStyle', 'None'); % plot with defaults
if nargin >= 8
    set(h, varargin{:}) % if stuff is specified, change it to match those specs
end
if nargout == 1
    varargout(1) = {h};
end
