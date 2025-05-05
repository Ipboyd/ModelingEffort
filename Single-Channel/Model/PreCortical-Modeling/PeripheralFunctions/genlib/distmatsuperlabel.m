function distmatsuperlabel(varargin)

if ishandle(varargin{1})
	hDistMatAxes = varargin{1};
	labels = varargin{2};
	argNum = 3;
else
	hDistMatAxes = gca;
	labels = varargin{1};
	argNum = 2;
end

figure(get(hDistMatAxes, 'Parent'));
hLabelAxes = axes;
oldUnits = get(hDistMatAxes, 'Units');

units = 'inches';
labelTickLen = 0.2;
labelMargin = 0.2;
labelOnBottom = 1;
labelOnLeft = 1;
lineStyle = '-';
labelAlignment = 'out';
xRotation = 'off';
yRotation = 'on';
fontSize = get(hDistMatAxes, 'FontSize');
doXLabel = 'on';
doYLabel = 'on';
yDir = get(hDistMatAxes, 'YDir');

xHandles = [];
yHandles = [];

for argNum = argNum:2:nargin
	switch lower(varargin{argNum})
		case 'units'
			units = varargin{argNum + 1};
		case 'labelticklen'
			labelTickLen = varargin{argNum + 1};
		case 'labelmargin'
			labelMargin = varargin{argNum + 1};
		case 'labelposition'
			if any(varargin{argNum + 1} == 't')
				labelOnBottom = 0;
			end
			if any(varargin{argNum + 1} == 'b')
				labelOnBottom = 1;
			end
			if any(varargin{argNum + 1} == 'r')
				labelOnLeft = 0;
			end
			if any(varargin{argNum + 1} == 'l')
				labelOnLeft = 1;
			end
		case 'linestyle'
			lineStyle = varargin{argNum + 1};
		case 'labelalignment'
			labelAlignment = lower(varargin{argNum + 1});
		case 'xrotation'
			xRotation = varargin{argNum + 1};
		case 'yrotation'
			yRotation = varargin{argNum + 1};
		case 'fontsize'
			fontSize = varargin{argNum + 1};
		case 'doxlabel'
			doXLabel = varargin{argNum + 1};
		case 'doylabel'
			doYLabel = varargin{argNum + 1};
		otherwise
			error('Argument %0.0f not understood', argNum)
	end
end

set([hDistMatAxes, hLabelAxes], 'Units', units)
distMatAxesPos = get(hDistMatAxes, 'position');
% Try to account for a manual DataAspectRatioMode
if strcmpi(get(hDistMatAxes, 'PlotBoxAspectRatioMode'), 'Manual')
	plotBoxAspectRatio = get(hDistMatAxes, 'PlotBoxAspectRatio');
	ratio = plotBoxAspectRatio(2)/plotBoxAspectRatio(1);
	maxRatio = distMatAxesPos(4)/distMatAxesPos(3);
	if maxRatio > ratio
		newHeight = ratio*distMatAxesPos(3);
		distMatAxesPos(2) = distMatAxesPos(2) + (distMatAxesPos(4) - newHeight)/2;
		distMatAxesPos(4) = newHeight;
	elseif maxRatio < ratio
		newWidth = ratio\distMatAxesPos(4);
		distMatAxesPos(1) = distMatAxesPos(1) + (distMatAxesPos(3) - newWidth)/2;
		distMatAxesPos(3) = newWidth;
	end
end
labelAxesPos = [distMatAxesPos(1:2) - [labelOnLeft, labelOnBottom].*labelTickLen, distMatAxesPos(3:4) + labelTickLen];
set(hLabelAxes, 'Position', labelAxesPos, 'XLim', labelAxesPos(1) + [0, labelAxesPos(3)], 'YLim', labelAxesPos(2) + [0, labelAxesPos(4)])

if strcmp(xRotation, 'on')
	xRotation = 1;
else
	xRotation = 0;
end
if strcmp(yRotation, 'on')
	if labelOnLeft
		yRotation = 1;
	else
		yRotation = 3;
	end
else
	yRotation = 0;
end

if strcmp(doXLabel, 'off')
	doXLabel = 0;
else
	doXLabel = 1;
end
if strcmp(doYLabel, 'off')
	doYLabel = 0;
else
	doYLabel = 1;
end

if yRotation
	yHorizontalAlignment = 'Center';
	switch(labelAlignment)
		case 'in'
			yVerticalAlignment = 'Top';
		case 'out'
			yVerticalAlignment = 'Bottom';
		case {'middle', 'center'}
			yVerticalAlignment = 'Middle';
	end
else
	yVerticalAlignment = 'Middle';
	switch(labelAlignment)
		case 'in'
			if labelOnLeft
				yHorizontalAlignment = 'Left';
			else
				yHorizontalAlignment = 'Right';
			end
		case 'out'
			if labelOnLeft
				yHorizontalAlignment = 'Right';
			else
				yHorizontalAlignment = 'Left';
			end
		case {'middle', 'center'}
			yHorizontalAlignment = 'Center';
	end
end
if xRotation
	xVerticalAlignment = 'Middle';
	switch(labelAlignment)
		case 'in'
			if labelOnBottom
				xHorizontalAlignment = 'Left';
			else
				xHorizontalAlignment = 'Right';
			end
		case 'out'
			if labelOnBottom
				xHorizontalAlignment = 'Right';
			else
				xHorizontalAlignment = 'Left';
			end
		case {'middle', 'center'}
			xHorizontalAlignment = 'Center';
	end
else
	xHorizontalAlignment = 'Center';
	switch(labelAlignment)
		case 'in'
			if labelOnBottom
				xVerticalAlignment = 'Bottom';
			else
				xVerticalAlignment = 'Top';
			end
		case 'out'
			if labelOnBottom
				xVerticalAlignment = 'Top';
			else
				xVerticalAlignment = 'Bottom';
			end
		case {'middle', 'center'}
			xVerticalAlignment = 'Middle';
	end
end

numTicks = length(labels) + 1;

xLabelTick = linspace(distMatAxesPos(1), distMatAxesPos(1) + distMatAxesPos(3), numTicks);
yLabelTick = linspace(distMatAxesPos(2), distMatAxesPos(2) + distMatAxesPos(4), numTicks);

repInds = meshgrid(1:numTicks, 1:3);
repInds = repInds(:);

xx = xLabelTick(repInds);
xy(1:3:numTicks*3) = labelAxesPos(2);
xy(2:3:numTicks*3) = labelAxesPos(2) + labelAxesPos(4);
xy(3:3:numTicks*3) = NaN;

yx(1:3:numTicks*3) = labelAxesPos(1);
yx(2:3:numTicks*3) = labelAxesPos(1) + labelAxesPos(3);
yx(3:3:numTicks*3) = NaN;
yy = yLabelTick(repInds);

xy(1:3:numTicks*3) = labelAxesPos(2) + ~labelOnBottom*distMatAxesPos(4);
xy(2:3:numTicks*3) = distMatAxesPos(2) + ~labelOnBottom*labelAxesPos(4);
yx(1:3:numTicks*3) = labelAxesPos(1) + ~labelOnLeft*distMatAxesPos(3);
yx(2:3:numTicks*3) = distMatAxesPos(1) + ~labelOnLeft*labelAxesPos(3);


xTextX = xLabelTick(1:end - 1) + diff(xLabelTick)/2;
xTextY = repmat(distMatAxesPos(2) - labelOnBottom*labelMargin + ~labelOnBottom*(distMatAxesPos(4) + labelMargin), 1, numTicks - 1);
yTextX = repmat(distMatAxesPos(1) - labelOnLeft*labelMargin + ~labelOnLeft*(distMatAxesPos(3) + labelMargin), 1, numTicks - 1);
yTextY = yLabelTick(1:end - 1) + diff(yLabelTick)/2;

set(gcf, 'CurrentAxes', hLabelAxes)
axis off
hold on
xHandles = [xHandles; plot(xx, xy, 'LineWidth', get(hDistMatAxes, 'LineWidth'), 'Color', get(hDistMatAxes, 'XColor'), 'LineStyle', lineStyle, 'Clipping', 'Off')];
yHandles = [yHandles; plot(yx, yy, 'LineWidth', get(hDistMatAxes, 'LineWidth'), 'Color', get(hDistMatAxes, 'YColor'), 'LineStyle', lineStyle, 'Clipping', 'Off')];
hold off


if iscell(labels)
	if strcmpi(yDir, 'reverse')
		yLabels = fliplr(labels);
	else
		yLabels = labels;
	end
	xHandles = [xHandles; text(xTextX, xTextY, labels, 'FontName', get(hDistMatAxes, 'FontName'), 'FontSize', fontSize,...
		'Rotation', xRotation*90, 'VerticalAlignment', xVerticalAlignment, 'HorizontalAlignment', xHorizontalAlignment)];
	yHandles = [yHandles; text(yTextX, yTextY, yLabels, 'FontName', get(hDistMatAxes, 'FontName'), 'FontSize', fontSize,...
		'Rotation', yRotation*90, 'VerticalAlignment', yVerticalAlignment, 'HorizontalAlignment', yHorizontalAlignment)];
end
if ischar(labels)
	if size(labels, 1) == 1
		labels = labels.';
	end
	if strcmpi(yDir, 'reverse')
		yLabels = flipud(labels);
	else
		yLabels = labels;
	end
	for ii = 1:numTicks - 1
		xHandles = [xHandles; text(xTextX(ii), xTextY(ii), labels(ii,:), 'FontName', get(hDistMatAxes, 'FontName'), 'FontSize', fontSize,...
			'Rotation', xRotation*90, 'VerticalAlignment', xVerticalAlignment, 'HorizontalAlignment', xHorizontalAlignment)];
		yHandles = [yHandles; text(yTextX(ii), yTextY(ii), yLabels(ii,:), 'FontName', get(hDistMatAxes, 'FontName'), 'FontSize', fontSize,...
			'Rotation', yRotation*90, 'VerticalAlignment', yVerticalAlignment, 'HorizontalAlignment', yHorizontalAlignment)];
	end
end

if ~doXLabel
	delete(xHandles)
end
if ~doYLabel
	delete(yHandles)
end

set([hDistMatAxes, hLabelAxes], 'Units', oldUnits)
% children = get(gcf, 'Children');
% indDistMatAxes = find(children == hDistMatAxes);
% if hDistMatAxes == children(end)
% 	set(gcf, 'Children', [children(2:indDistMatAxes); hLabelAxes])
% else
% 	set(gcf, 'Children', [children(2:indDistMatAxes); hLabelAxes; children(indDistMatAxes + 1:end)])
% end
set(gcf, 'CurrentAxes', hDistMatAxes)