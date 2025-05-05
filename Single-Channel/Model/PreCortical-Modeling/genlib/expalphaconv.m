function out = expalphaconv(x, taue, taua, fs)
% function out = expalphaconv(x,taue,taua,fs)
%   OUT = EXPALPHACONV(X,TAUE,TAUA,FS) convolves the matrix of column 
%   vectors X with an exponential kernel with time constant TAUE minus an 
%   alpha kernel with time constant TAUA. The exp-alpha kernel integrates
%   to zero area, useful for simulating a delayed-inhibition EPSP.
%
%   It is advantageous to use EXPALPHACONV instead of CONV or FFTCONV
%   because an exponential function minus an alpha function is a two-zero 
%   three-pole filter, requiring only 5*N operations.
%

% 
w = exp(-1/taue/fs);
y = exp(-1/taua/fs);
C = taue/taua/taua/fs;
out = filter([(1) (-y*(C+2)) (y*(y+C*w))],[(1) (-2*y-w) (y*(y+2*w)) (-y*y*w)],x);