function newmat = reorder_distmat(mat, order, varargin)
%REORDER_DISTMAT Reorders a distance matrix according to an order vector.
%   REORDER_DISTMAT(MAT, ORDER) where MAT is the distance matrix to be
%   rearranged and ORDER is the order of the blocks. Each block is n-by-n
%   elements, where n = length(MAT)/length(ORDER). As a result, the side
%   length of MAT must be an integer multiple of the length of ORDER.

% newmat = reorder_distmat(mat, order, undo)
% A function that allows a distance matrix to be rearranged for calculating
% the percent correct.  If the third argument is absent or 0, it rearranges
% the matrix according to order.  If it is nonzero, then it undoes a
% reordering that was done with that order matrix.

if nargin == 2
    reverse = 0;
else
    reverse = logical(varargin{1});
end

len = length(mat);
numstims = length(order);
numtrials = len/numstims;
numlayers = size(mat, 3);
inds = zeros(1,len);

if mod(numtrials,1)
    error('The length of order resulted in a non-integer number of trials.')
end

for i = 1:numstims
	if order(i)
	    inds(((i - 1)*numtrials + 1):i*numtrials) = ((order(i) - 1)*numtrials + 1):order(i)*numtrials;
	end
end
inds(~inds) = [];
newlen = length(inds);
if length(unique(inds)) < newlen
	warning('There are repeated entries in your order vector.')
end

if ~reverse
newmat = zeros(newlen, newlen, numlayers);
    for layer = 1:numlayers
        newmat(:,:,layer) = mat(inds,inds,layer);
    end
else
	if length(inds) < len
		error('Order vector cannot include zeros when the reverse flag is on.')
	else
		for layer = 1:size(mat,3)
			newmat(inds,inds,layer) = mat(:,:,layer);
		end
	end
end