function plotsize(width, height, h, units)
% PLOTSIZE(WIDTH, HEIGHT, H, UNITS)  [DEPRICATED]
%    WIDTH and HEIGHT are the size in UNITS. H is handle to figure (defaults to
%    gcf). For proper size display onscreen, you need to use the command,
%    SET(0, 'ScreenPixelsPerInch', DPI), where DPI is the number of pixels per
%    inch of your monitor (which you use a ruler to calculate).  For a 19-inch
%    monitor with 1280x1024 resolution, this value is approximately 86.23.
%    
%    PLOTSIZE is being replaced by FIGURESIZE.
%    

if nargin < 2
	error('At least two arguments are needed.')
end
if nargin < 3
    h = gcf;
end
if nargin < 4
	units = 'inches';
	if isempty(h)
		h = gcf;		
	end
end

figuresize(width, height, h, units);
