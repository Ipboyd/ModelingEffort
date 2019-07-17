% function out = stochsyn(x,tau,fs,p)
%   OUT = STOCHSYN(X,TAU,FS,P) stochastically reduces the spikes in each 
%   column vector of matrix X by iteratively finding spikes (denoted by 
%   integer spike counts 1, 2, etc.) and deleting them with probabilities 
%   determined by the sum of decaying exponentials, scaled by P with time 
%   constant TAU, placed at the locations of all previous spikes.
%
%   e.g. A run of stochsyn(repmat([1;0;0;1;1;2;1;0;0],1,4),1,1,0.5) yields:
%
%        ans =
%
%             1     1     1     1
%             0     0     0     0
%             0     0     0     0
%             1     1     1     1
%             1     1     0     0
%             2     0     2     1
%             0     1     0     1
%             0     0     0     0
%             0     0     0     0
%
% NOTE: All inputs must be double datatype, e.g. if X is a logical array,
%         pass the function double(X) instead of X.
%   
