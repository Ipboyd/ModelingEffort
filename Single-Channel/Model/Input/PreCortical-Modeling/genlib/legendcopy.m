function [hleg1, hobj1, hplot1, strings1] = legendcopy(haxes0, haxes1)
% function [hleg1, hobj1, hplot1, strings1] = legendcopy(haxes0, haxes1)
%   Copies the legend from one axes (HAXES0) to another (HAXES1), returning the
%   same velues that LEGEND resturns for the new legend.
%
%   See also legend, figs2subplots.

% get info from source plot
axes(haxes0)
children0 = get(haxes0, 'children');

% get info from destination plot
axes(haxes1)
children1 = get(haxes1, 'children');

% wimpy error checking
if length(children0) ~= length(children1)
	error('The source and destination axes must have the same number of children.')
end

% get info from source and find out which children of the source plot to
% which each legend entry refers
axes(haxes0)
[hleg0, hobj0, hplot0, strings0] = legend;
location = get(hleg0, 'location');
position = get(hleg0, 'position');
hleg0 = legend(hplot0, strings0);
children0 = get(haxes0, 'children');
inds = [];
for ii = 1:length(hplot0)
	inds = [inds find(children0 == hplot0(ii))];
end

% make destination legend
axes(haxes1)
hplot1 = children1(inds);
[hleg1, hobj1, hplot1, strings1] = legend(hplot1, strings0);
set(hleg1, 'tag', 'legend')

% properly place legend (location is easy, position is tougher)
if ~strcmp(location, 'none')
	set([hleg0 hleg1], 'location', location)
else
	apos0 = get(haxes0, 'position');
	apos1 = get(haxes1, 'position');
	amid0 = apos0(1:2) + apos0(3:4)/2;
	amid1 = apos1(1:2) + apos1(3:4)/2;
	lpos0 = position;
	lmid0 = lpos0(1:2) + lpos0(3:4)/2;
	lpos1 = get(hleg1, 'position');
	shift0 = lmid0 - amid0;
	scale = apos1(3:4)./apos0(3:4);
	shift = shift0.*scale;
	lmid1 = amid1 + shift;
	lpos1 = [lmid1 - lpos1(3:4)/2, lpos1(3:4)];
	
	set(hleg0, 'position', lpos0)
	set(hleg1, 'position', lpos1)
end