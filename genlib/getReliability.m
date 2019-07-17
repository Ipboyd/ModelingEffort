function [r,mat] = getReliability(inMat,sigma,fs)
% function r = getReliability(inmat,sigma)
%   R = GETRELIABILITY(INMAT,SIGMA) convolves the matrix of column vectors 
%   INMAT with a Gaussian kernel with standard deviation SIGMA (in units of
%   time) and then computes a correlation-based measure of spike reliability. 
%   INMAT can also be a cell array of spike times, which will be converted 
%   to a spike matrix with sample rate 1000. (SIGMA in this case is assumed
%   to be a time, which will automatically be converted to a number of 
%   samples.) If you pass in the variable FS as a third argument and you
%   pass in a cell array of spike times, conversions of INMAT and SIGMA
%   will use the specified sample rate FS.
%
%   [R,MAT] = GETRELIABILITY(...) will also return the matrix of normalized
%   inner products used to compute the total reliability.
%
%   This function depends on both normconv and upperCorrSum mex-files.
%

if(nargin < 3 || isempty(fs))
    fs = 1000;
end
if iscell(inMat)
	inMat = st2sm(inMat, fs);
    sigma = max(round(sigma*fs),1);
end

[len Ntrials] = size(inMat);
if(Ntrials < 2 || len < 1)
    warning('Too few trials to compute reliability; returning -1 (and []).');
    r = -1;
    mat = [];
    return;
end

%% This is an equivalent operation that doesn't use the mexed upperCorrSum
% convMat = normconv(inMat,sigma);
% matNorms = max(sqrt(sum(convMat.^2)),min(0.1/len,0.00001));
% r = 0;
% for ii = 1:Ntrials
%     rtemp = 0;
%     for jj = ii+1:Ntrials
%         rtemp = rtemp + (convMat(:,ii)'*convMat(:,jj)) / matNorms(jj);
%     end
%     r = r + rtemp/matNorms(ii);
% end
% r = r*2 / (Ntrials*(Ntrials-1));

[r,mat] = upperCorrSum(normconv(inMat,sigma));
