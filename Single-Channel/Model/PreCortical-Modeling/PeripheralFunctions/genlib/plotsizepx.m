function plotsize(w, h, f)
% plotsize(w, h, f)
% w is width in inches
% h is height in inches
% f is handle to figure (defaults to gcf)

if nargin == 2
    f = gcf;
end

set(f, 'units', 'pixels', 'color', [1 1 1])
set(f, 'position', [32 32 w h])
% set(f, 'resize', 'off')

a = get(f, 'children');

for i = a;
%     set(i, 'fontname', 'Tahoma')
%     pos = get(i, 'position');
%     ti = get(i, 'tightinset');
%     out = get(i, 'outerposition');
%     set(i, 'position', [out(1:2) + ti(1:2) + .02, out(3:4) - ti(3:4) - .04]);
end

figure(f)