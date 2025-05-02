function J = jetblack(m)
%JET    Variant of HSV
%   JETBLACK(M), a variant of JET(M), has pure black as it's low-value
%   color, rather than [0, 0, 0.5]. It is otherwise the same as JET.
%
%   See also JET, HSV, HOT, PINK, FLAG, COLORMAP, RGBPLOT.

%   Copyright 1984-2004 The MathWorks, Inc.
%   $Revision: 5.7.4.2 $  $Date: 2005/06/21 19:31:40 $

if nargin < 1
   m = size(get(gcf,'colormap'),1);
end
n = ceil(m/4);
u = [(1:1:n)/n ones(1,n-1) (n:-1:1)/n]';
ub = [(0:2:n - 1)/n ones(1,n-1) (n:-1:1)/n]';
ub = [zeros(length(u) - length(ub), 1); ub];
g = ceil(n/2) - (mod(m,4)==1) + (1:length(u))';
r = g + n;
b = g - n;
g(g>m) = [];
r(r>m) = [];
b(b<1) = [];
J = zeros(m,3);
J(r,1) = u(1:length(r));
J(g,2) = u(1:length(g));
J(b,3) = ub(end-length(b)+1:end);
