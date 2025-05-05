% function out = poisscount(LAMBDA)
%   OUT = POISSCOUNT(LAMBDA) generates random poisson spike counts
%   corresponding to means LAMBDA, where LAMBDA is an NxM matrix of mean
%   values. There is a function in the statistics toolbox which does this
%   same thing (poissrnd), but it generates the same distributions over and
%   over again.
%