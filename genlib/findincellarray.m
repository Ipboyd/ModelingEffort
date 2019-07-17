function indices = findincellarray(cells, key, compare)

%FINDCELL Finds cells of a cell array that satisfy a condition.
%   FINDCELL(CELLS, KEY) where key is a scalar or a string searches
%   for cells in CELLS that are equal to KEY and returns these indices.
%   Where KEY is a numerical array, it must have the same number of
%   elements as CELLS, and it will return the indices for which
%   corresponding elements in KEY and cells in CELLS were of equal value.
%   Where KEY is a cell array, an element-by-element comparison will be
%   made between cells, where differing types are false.
%
%   FINDCELL(CELLS, KEY, COMPARE) behaves as above except in the
%   case of numeric elements, which when encountered will be compared
%   according to the string COMPARE, with CELLS on the left-hand side and
%   KEY on the right-hand side. FINDCELL(CELLS, KEY, '==') is
%   equivalent to FINDCELL(CELLS, KEY).
%
%   See also find, ind2sub.

% Ross Maddox, 07 August 2007


if nargin == 2
    compare = [];
end

cells = cells(:);

if isnumeric(key)
    indices = findnumeric(cells, key, compare);
elseif ischar(key)
    indices = findchar(cells, key);
elseif iscell(key)
    indices = findcell(cells, key, compare);
end


%% compare numbers
function indices = findnumeric(cells, key, compare)

if isempty(compare)
    compare = '==';
end

numcells = length(cells);
results = false(numcells, 1);

if length(key) == 1
    key = repmat(key, numcells, 1);
end

for i = 1:numcells
    if isnumeric(cells{i})
        if eval(sprintf('%d %s %d', cells{i}, compare, key(i)))
            results(i) = true;
        end
    end
end

indices = find(results);


%% compare strings
function indices = findchar(cells, key)

numcells = length(cells);
results = false(numcells, 1);

if length(key) == 1
    key = repmat(key, numcells, 1);
end

for i = 1:numcells
    if ischar(cells{i})
        if strcmp(cells{i}, key)
            results(i) = true;
        end
    end
end

indices = find(results);


%% compare cell arrays
function indices = findcell(cells, key, compare)

if isempty(compare)
    compare = '==';
end

numcells = length(cells);
results = false(numcells, 1);

for i = 1:numcells
    if isnumeric(cells{i}) && isnumeric(key{i})
        if eval(sprintf('%d %s %d', cells{i}, compare, key{i}))
            results(i) = true;
        end
    elseif ischar(cells{i}) && ischar(key{i})
        if strcmp(cells{i}, key{i})
            results(i) = true;
        end
    end
end

indices = find(results);