function expandplots(varargin)

% EXPANDPLOTS(FIGHANDLE, SIZE, OPTIONS)
%
% Takes all the axes in a figure (fighandle, defaults to gcf) and expands them
% to a specified margin.

% This needs to be extended to utilize this: get(h, 'TightInset') + get(h, 'TightInset'), RKM May, 09


units = 'normalized';
method = 'matlab';
source = 'paper';
usetightinset = 0;

if nargin == 0
	input = 0;
% 	error('EXPANDPLOTS needs at least one input')
elseif nargin == 1;
	input = varargin{1};
else
	for ii = 1:nargin
		if length(varargin{ii}) == 1 && ishandle(varargin{ii}(1)) && isequal(varargin{ii}, round(varargin{ii})) && ~exist('hfig', 'var') && varargin{ii} ~= 0
			hfig = varargin{ii};
		elseif isnumeric(varargin{ii}) && ~exist('input', 'var')
			input = varargin{ii};
		elseif ischar(varargin{ii})
			switch lower(varargin{ii})
				case 'normalized'
				case 'absolute'
					units = 'absolute';
				case 'matlab'
				case 'corners'
					method = 'corners';
				case 'margin'
					method = 'margin';
				case 'paper'
				case 'screen'
					source = 'screen';
				case 'tightinset'
					usetightinset = 1;
				otherwise
					error('Argument %0.0f not understood', ii)
			end
		else
			error('Argument %0.0f not understood', ii)
		end
	end
end
if ~exist('hfig', 'var')
	hfig = gcf;
end

switch units
	case 'absolute'
		figsize = get(hfig, 'papersize');
	case 'normalized'
		figsize = [1 1];
end

if length(input) == 1
	input(2) = input(1);
end
if length(input) == 2
	switch method
		case 'matlab'
			input = [input, figsize - 2*input];
		case 'corners'
			input = [input, figsize - input];
		case 'margin'
			input = [input, input];
	end
end
if length(input) == 4
	switch method
		case 'matlab'
			newbound = [input([1 2]), input([1 2]) + input([3 4])];
		case 'corners'
			newbound = input;
		case 'margin'
			newbound = [input([1 2]), figsize - input([3 4])];
	end
end
newbound = newbound./figsize([1 2 1 2]);

if ~exist('newpos', 'var')
	newpos = 0.1;
end
if length(newpos) == 1
	newpos = [newpos*[1 1], 1 - 2*newpos*[1 1]];
end
if length(newpos) == 2
	newpos = [newpos, 1 - 2*newpos];
end

% hfig = gcf;

hall = get(hfig, 'children');
types = get(hall, 'type');
tags = get(hall, 'tag');

if ~iscell(types)
	types = {types};
	tags = {tags};
end
hplots = hall(intersect(findincellarray(types, 'axes'), findincellarray(tags, '')));
numplots = length(hplots);

positions = get(hplots, 'position');
tightinsets = get(hplots, 'tightinset');
if ~iscell(positions)
	positions = {positions};
end
if ~iscell(tightinsets)
	tightinsets = {tightinsets};
end
position = zeros(numplots, 4);
tightinset = zeros(numplots, 4);
for ii = 1:numplots
	position(ii,:) = positions{ii};
	tightinset(ii,:) = tightinsets{ii};
end

bounds = [position(:,1:2), position(:,1:2) + position(:,3:4)];
if numplots > 1
	absbound = [min(bounds(:,1:2)) max(bounds(:,3:4))];
else
	absbound = bounds;
end
scalefactor = [diff(newbound([1 3]))/diff(absbound([1 3])), diff(newbound([2 4]))/diff(absbound([2 4]))];

newpositions = (position + usetightinset*tightinset.*repmat([1, 1, -1, -1], numplots, 1)...
	- repmat([absbound(1:2) 0 0], numplots, 1)).*repmat(scalefactor([1 2 1 2]), numplots, 1) + repmat([newbound(1:2) 0 0], numplots, 1);

for ii = 1:numplots
	set(hplots(ii), 'position', newpositions(ii,:))
end