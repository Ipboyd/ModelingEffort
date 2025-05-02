% function out = upperCorrSum(x)
%   OUT = UPPERCORRSUM(X) calculates the normalized sum of the 
%   inner products of column vectors of the matrix X using a 
%   mex file as:
%
%	 matNorms = max(sqrt(sum(x.^2)),min(0.1/len,0.00001));
%	 r = 0;
%	 for ii = 1:Ntrials
%	     rtemp = 0;
%	     for jj = ii+1:Ntrials
%	         rtemp = rtemp + (convMat(:,ii)'*convMat(:,jj)) / matNorms(jj);
%	     end
%	     r = r + rtemp/matNorms(ii);
%	 end
%	 r = r*2 / (Ntrials*(Ntrials-1));
%