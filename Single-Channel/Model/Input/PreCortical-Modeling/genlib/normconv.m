% function out = normconv(x,sigma,pad)
%   OUT = NORMCONV(X,SIGMA,PAD) convolves the matrix of column vectors X
%   with an approximation to a normal kernel. It is /not/ exact, but it is 
%   symmetric and normal-looking. The normal-like kernel is normalized to
%   have area=1, mean zero, and standard deviation SIGMA in samples. The
%   optional parameter PAD is the number of zeros to pad the signal with
%   for computation; 3*SIGMA is the default and usually shouldn't need to
%   be changed.
%   
%   NOTE: All inputs must be double datatype, e.g. if X is a logical array,
%         pass the function double(X) instead of X.
%   
