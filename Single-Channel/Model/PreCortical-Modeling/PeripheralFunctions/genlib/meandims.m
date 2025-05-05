function m = meandims(x, dims, nanFlag, squeezeFlag)
%MEANDIMS Compute mean along specified dimensions.
%   MEANDIMS(X,DIMS,NANFLAG,SQUEEZEFLAG) computes the mean of X along 
%   the list of dimensions DIMS. Only the first two arguments are required.
%   If NANFLAG is true, it will return the mean of the non-NaN numbers;
%   if flase (default), it will return NaN if there is a NaN in the 
%   calculation. If SQUEEZEFLAG is true (default is false), it will compact
%   the output so that there are no singleton dimensions. This is
%   equivalent to running SQUEEZE(MEANDIMS(...)).
%
%   See also mean, nanmean, squeeze.

if nargin < 2 || isempty(x) || isempty(dims)
	error('Must input X and DIMS arguments.')
end
if nargin < 3 || isempty(nanFlag)
	nanFlag = false; % normal mean() behavior
end
if nargin < 4 || isempty(squeezeFlag)
	squeezeFlag = false; % normal mean() behavior
end

if any(mod(dims, 1))
	error('DIMS arugument must be all integers.')
end

nDims = numel(dims);
m = x;
if ~nanFlag
	for dimNum = 1:nDims
		m = mean(m, dims(dimNum));
	end
else
	for dimNum = 1:nDims
		m = nanmean(m, dims(dimNum));
	end
end
if squeezeFlag
	m = squeeze(m);
end