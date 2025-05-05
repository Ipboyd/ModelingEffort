function setlabelpos(varargin)
% setlabelpos(h_axes, x, y, t, options)
% setlabelpos(x, y, t, options) uses gca for h_axes

units = 'absolute';
ref = 'out';

if nargin > 3 && isnumeric(varargin{4})
	hax = varargin{1};
	xval = varargin{2};
	yval = varargin{3};
	tval = varargin{4};
	iistart = 5;
else
	hax = gca;
	xval = varargin{1};
	yval = varargin{2};
	tval = varargin{3};
	iistart = 4;
end
for ii = iistart:nargin
	switch varargin{ii}
		case 'normalized'
			units = varargin{ii};
		case 'absolute'
			units = varargin{ii};
		case 'in'
			ref = varargin{ii};
		case 'out'
			ref = varargin{ii};
		case 'middle'
			ref = varargin{ii};
		case 'center'
			ref = 'middle';
		otherwise
			error('Did not understand argument %0.0f', ii);
	end
end

for ii = 1:length(hax)
	hx = get(hax(ii), 'xlabel');
	hy = get(hax(ii), 'ylabel');
	ht = get(hax(ii), 'title');
	oldUnits = {get(hx, 'Units'), get(hy, 'Units'), get(ht, 'Units')}; % added 2/26/09
	set([hx hy ht], 'units', 'normalized')
	xpos = get(hx, 'position');
	ypos = get(hy, 'position');
	tpos = get(ht, 'position');
	
	axpos = get(hax(ii), 'position');
	axsize_fig = axpos([3 4]);
	xlim = get(hax(ii), 'xlim');
	ylim = get(hax(ii), 'ylim');
	axsize_ax = [diff(xlim) diff(ylim)];

	switch units
		case 'absolute'
			figpos = get(get(hax(ii), 'parent'), 'paperposition');
		case 'normalized'
			figpos = get(get(hax(ii), 'parent'), 'position');
	end
	figsize = figpos([3 4]);

	if ~isempty(xval)
		xpos(2) = -xval/figsize(2)/axsize_fig(2);
		set(hx, 'position', xpos)
	end
	if ~isempty(yval)
		ypos(1) = -yval/figsize(1)/axsize_fig(1);
		set(hy, 'position', ypos)
	end
	if ~isempty(tval)
		tpos(2) = 1 + tval/figsize(2)/axsize_fig(2);
		set(ht, 'position', tpos)
	end
% 	set(hx, 'units', oldUnits{1}) % restore to old units (default is 'data')
% 	set(hy, 'units', oldUnits{2})
% 	set(ht, 'units', oldUnits{3})

	switch ref
		case 'in'
			set(hx, 'verticalalign', 'top')
			set([hy ht], 'verticalalign', 'bottom')
		case 'out'
			set(hx, 'verticalalign', 'bottom')
			set([hy ht], 'verticalalign', 'top')
		case 'middle'
			set([hx hy ht], 'verticalalign', 'middle')
	end
end