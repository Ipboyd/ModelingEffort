function J = jetmod(m)
%JETMOD    Variant of JET
%   JETMOD(M) is a variant of JET(M) intended for coloring lines
%   and markers, rather than images or surfaces.
%
%   JET(M), a variant of HSV(M), is an M-by-3 matrix containing
%   the default colormap used by CONTOUR, SURF and PCOLOR.
%   The colors begin with dark blue, range through shades of
%   blue, cyan, green, yellow and red, and end with dark red.
%   JET, by itself, is the same length as the current figure's
%   colormap. If no figure exists, MATLAB creates one.
%
%   See also JET, HSV, HOT, PINK, FLAG, COLORMAP, RGBPLOT.

%   Copyright 1984-2004 The MathWorks, Inc.
%   $Revision: 5.7.4.2 $  $Date: 2005/06/21 19:31:40 $

if nargin < 1
   m = size(get(gcf,'colormap'),1);
end
% m_adj = ceil(8/6*(m - 1));
% n = ceil(m_adj/4);
% u = [(1:1:n)/n ones(1,n-1) (n:-1:1)/n]';
% g = ceil(n/2) - (mod(m_adj,4)==1) + (1:length(u))';
% r = g + n;
% b = g - n;
% g(g>m_adj) = [];
% r(r>m_adj) = [];
% b(b<1) = [];
% J = zeros(m_adj,3);
% J(r,1) = u(1:length(r));
% J(g,2) = u(1:length(g));
% J(b,3) = u(end-length(b)+1:end);

bounds = ceil(m*[0.08 0.3 0.7 0.95]);

J = zeros(m,3);
J(bounds(2):bounds(3),1) = linspace(0, 1, diff(bounds([2 3])) + 1);
J(bounds(3):bounds(4),1) = 1;

J(bounds(4):end,1) = linspace(1, 0.8, m - bounds(4) + 1);

J(bounds(1):bounds(2),2) = linspace(0, 1, diff(bounds([1 2])) + 1);
J(bounds(2):bounds(3),2) = 1;
J(bounds(3):bounds(4),2) = linspace(1, 0, diff(bounds([3 4])) + 1);

J(1:bounds(1),3) = linspace(0.8, 1, bounds(1));
J(bounds(1):bounds(2),3) = 1;
J(bounds(2):bounds(3),3) = linspace(1, 0, diff(bounds([2 3])) + 1);

p = 3;
J(:,1:2) = J(:,1:2) - .5^p*repmat(min(J(:,1),J(:,2)),1,2).^p;
J(bounds(1):bounds(4),[1 3]) = J(bounds(1):bounds(4),[1 3]).^1.2;
% J(:,1:2) = J(:,1:2) - 0.2*repmat(min(J(:,1),J(:,2)),1,2).^3;

% J = J(1 + floor(m/8):m + floor(m/8),:);