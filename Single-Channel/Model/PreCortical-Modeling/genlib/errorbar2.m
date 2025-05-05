function varargout = errorbar2(x, y, e, varargin)

% H = ERRORBAR2(X, Y, E, varargin)
%   varargin is all the line-formatting arguments you would give the plot function
%   Unlike errorbar, this only plots the bars, not the line connecting them.  Also,
%   you can have several series in Y and E (each column is a series).

if all(size(x) > 1)
	numSeries = size(x, 2);
else
	x = x(:);
	numSeries = 1;
end

h = [];
for seriesNum = 1:numSeries
	xb = x(ceil(1/3:1/3:length(x)), seriesNum);
	yb = zeros(length(y)*3, size(y, 2));
	yb(1:3:end, :) = y(:, seriesNum) - e(:, seriesNum);
	yb(2:3:end, :) = y(:, seriesNum) + e(:, seriesNum);
	yb(3:3:end, :) = nan;

	h = [h plot(xb, yb',varargin{:})]; %#ok<AGROW>
end
if nargout == 1
	varargout(1) = {h};
end