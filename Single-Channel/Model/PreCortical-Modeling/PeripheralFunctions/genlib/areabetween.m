function area = areabetween(x1,x2,y1,y2)
% function out = areabetween(x1,x2,y1,y2)
%   AREA = AREABETWEEN(...) calculates the area between two functions (i.e.
%   for every Xi there is exactly one Yi) defined by the sorted pairings
%   (X1,Y1) and (X2,Y2). If the x-coordinates X1 and X2 are not the same,
%   linear interpolation is performed. If X1 and X2 are the same, you can
%   also call AREABETWEEN(X1,[],Y1,Y2).
%
% 2008/08/07 Eric Larson edlarson@bu.edu
% 

% Check inputs, interpolate X- and Y-coordinates if X's don't match
if(isempty(x2))
    x = x1(:);
    if(length(x) ~= length(y1) || length(x) ~= length(y2))
        error('x1 must be of the same length as both y1 and y2 when x2 = []');
    end
else
    if(length(x1) ~= length(y1))
        error('x1 must be of the same length as y1');
    elseif(length(x2) ~= length(y2))
        error('x2 must be of the same length as y2');
    end
    if(~isequal(x1,x2))
        x = unique([x1(:);x2(:)]);
        y1 = interp1(x1,y1(:),x);
        y2 = interp1(x2,y2(:),x);
    else
        x = x1(:);
    end
end
y1 = y1(:);
y2 = y2(:);

% Deal with curve crossings by adding points if necessary
crossinds = find(diff(y1>y2));
if(~isempty(crossinds))
    [xints,yints] = intersections(x(1:end-1),x(2:end),x(1:end-1),x(2:end),y1(1:end-1),y1(2:end),y2(1:end-1),y2(2:end));
    % This code was bombing out, so changed to below RKM 2012/09/07
	% oinds = ones(length(x1),1);
    % oinds(crossinds+1) = 2;
    % oinds = [cumsum(oinds);crossinds+(1:length(crossinds)).'];
    % x(oinds) = [x;xints(crossinds)];
    % y1(oinds) = [y1;yints(crossinds)];
    % y2(oinds) = [y2;yints(crossinds)];
	[x, order] = sort([x;xints(crossinds)]);
	y1 = [y1;yints(crossinds)];
	y1 = y1(order);
	y2 = [y2;yints(crossinds)];
	y2 = y2(order);
end

dy = abs(y1-y2);
dy = (dy(1:end-1)+dy(2:end))/2;
area = sum(diff(x).*dy);

function [xints,yints] = intersections(x1,x2,x3,x4,y1,y2,y3,y4)
d = ((x1-x2).*(y3-y4) - (x3-x4).*(y1-y2));
xints = ((x1.*y2-x2.*y1).*(x3-x4) - (x3.*y4-x4.*y3).*(x1-x2))./d;
yints = ((x1.*y2-x2.*y1).*(y3-y4) - (x3.*y4-x4.*y3).*(y1-y2))./d;

% % An equivalent (~10% slower) point-adding calculation can be performed as:
% if(~isempty(crossinds))
%     [xints,yints] = intersections(x(1:end-1),x(2:end),x(1:end-1),x(2:end),y1(1:end-1),y1(2:end),y2(1:end-1),y2(2:end));
%     [junk,sortinds] = sort([1:length(x) (crossinds+0.5).']);
%     x = [x;xints(crossinds)]; x = x(sortinds);
%     y1 = [y1;yints(crossinds)]; y1 = y1(sortinds);
%     y2 = [y2;yints(crossinds)]; y2 = y2(sortinds);
% end

% % An equivalent (~600% slower) area calculation can now be performed as:
% area = polyarea([x;flipud(x);],[max([y1 y2],[],2);flipud(min([y1 y2],[],2))]);
