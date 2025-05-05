function varargout = patchbar(h, patchProps, lineProps)

if ~exist('patchProps', 'var') || isempty(patchProps)
	patchProps = {'Color', 0.5*[1 1 1]};
elseif ~iscell(patchProps)
	patchProps = {patchProps};
end

if ~exist('lineProps', 'var') || isempty(lineProps)
	lineProps = {'Color', [0 0 0]};
elseif ~iscell(lineProps)
	lineProps = {lineProps};
end

xData = get(h, 'XData');
yData = get(h, 'YData');
xData = xData(:);
yData = yData(:);

len = length(xData);

inds = sort([1:3 6:4:len 7:4:len len]);
xx = xData(inds);
yy = yData(inds);

delete(h)
hPatch = patch(xx, yy, patchProps{:}, 'LineStyle', 'None');
hold on
hLine = line(xx, yy, lineProps{:});
hold off

if nargout >= 1
    varargout{1} = hPatch;
end
if nargout >= 2
    varargout{2} = hLine;
end