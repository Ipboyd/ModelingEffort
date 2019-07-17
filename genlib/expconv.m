% function out = expconv(x,tau,fs)
%   OUT = EXPCONV(X,TAU,FS) convolves the matrix of column vectors X with 
%   an exponential kernel with time constant tau. The exponential kernel is
%   not normalized---it is causal with initial value of unity.
%
%   It is advantageous to use EXPCONV instead of CONV or FFTCONV because an
%   exponential function is a one-pole filter---the necessary convolution 
%   can be calculated using only 2*N operations.
%
%   This mex-funtion calculates this operation 20X faster despite filter is
%   is also a mex-function (because it's specialized?):
%   
%      out = filter(1, [1 -exp(-1/fs/tau)], x);
%
%   NOTE: All inputs must be double datatype, e.g. if X is a logical array,
%         pass the function double(X) instead of X.
%   
