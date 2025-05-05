function out = max1dim(in,dim)
% function out = max1dim(in,dim)
%   OUT = MAX1DIM(IN,DIM) reduces the multi-dimensional matrix IN to a
%   vector by taking a series of maximizations MAX(in,[],...) across all
%   dimensions /except/ dim. This is useful for parameter space searches.
%

mystr = ['in'];
for ii = [1:dim-1 dim+1:ndims(in)]
    mystr = sprintf('max(%s,[],%d)',mystr,ii);
end; clear ii;

out = eval(['squeeze(' mystr ')']);
