function J = hotcold(m)
% HOTCOLD    Colormap for visualizing data centered at 0
%   HOTCOLD(M), is an M-by-3 matrix containing a colormap that is black at 0,
%   cold colors when negative and hot colors when positive HOTCOLD, by itself,
%   is the same length as the current figure's colormap. If no figure exists,
%   MATLAB creates one.
%
%   To center the figures colormap at 0, use the command:
%   set(h, 'CLim', [-1 1]*max(abs(get(h, 'CLim'))))
%
%   See also HSV, HOT, PINK, FLAG, COLORMAP, RGBPLOT.

%   rkmaddox@bu.edu

if nargin < 1
   m = size(get(gcf,'colormap'), 1);
end

zones = floor((m - 1)*(0:1/4:1) + 1);

r = zeros(m, 1);
g = zeros(m, 1);
b = zeros(m, 1);

r(zones(3):zones(4)) = linspace(0, 0.85, diff(zones([3 4])) + 1);
r(zones(4):zones(5)) = linspace(0.85, 1, diff(zones([4 5])) + 1);
% red does not go to 1 at zone(4) because it creates an unsightly pure [1 0 0]
% band when plotting smooth functions using pcolor.

g(zones(1):zones(2)) = linspace(1, 0, diff(zones([1 2])) + 1);
g(zones(4):zones(5)) = linspace(0, 1, diff(zones([4 5])) + 1);

b(zones(2):zones(3)) = linspace(1, 0, diff(zones([2 3])) + 1);
b(zones(1):zones(2)) = 1;

J = [r g b];