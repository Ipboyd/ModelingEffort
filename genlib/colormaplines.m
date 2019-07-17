function colormaplines(varargin)
% COLORMAPLINES  Apply a colormap to the lines in one or more axes.
%    COLORMAPLINES defaults to GCA's children and @JET
%    COLORMAPLINES(@COLORMAPFUNCTION) defaults to GCA's children
%    COLORMAPLINES(H, @COLORMAPFUNCTION) applies to H, or H's children
%    depending on if it is an array of line objects or an array of axes
%    objects, respectively. It will bomb if types are mixed.

switch nargin
	case 0
		h = get(gca, 'Children');
		h = h(strcmp(get(h, 'Type'), 'line'));
		cmFun = @hsv;
	case 1
		h = get(gca, 'Children');
		h = h(strcmp(get(h, 'Type'), 'line'));
		cmFun = varargin{1};
	case 2
		h = varargin{1};
		cmFun = varargin{2};
	otherwise
		error('COLORMAPLINES needs 0, 1, or 2 inputs.')
end

if strcmp(get(h(1), 'Type'), 'line')
	ha.hl = h;
	nAxes = 1;
elseif strcmp(get(h(1), 'Type'), 'axes')
	nAxes = length(h);
	for ii = 1:nAxes
		ha(ii).hl = get(h(ii), 'Children');
		ha(ii).hl = ha(ii).hl(strcmp(get(ha(ii).hl, 'Type'), 'line'));
	end
else
	error('COLORMAPLINES needs handles of ''type'' line or ''axes''')
end

for axesNum = 1:nAxes
	nLines = length(ha(axesNum).hl);
	colors = cmFun(nLines);
	for lineNum = 1:nLines
		set(ha(axesNum).hl(lineNum), 'Color', colors(lineNum,:));
	end
end