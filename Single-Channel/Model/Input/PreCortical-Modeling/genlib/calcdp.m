function [dp,stdJN] = calcdp(st,taus,numTargets,tempNum,fs,useMins,useTrials)
%CALCDP d' across stimuli for a spike matrix.
%   [DP,STDJN] = CALCDP(ST,TAUS,NUMTARGETS,TEMPNUM) calculates the average 
%   d' score for a set of spike time cells ST that is organized as 
%   st(#trials,#songs), with a vector of tau values TAUS, sample 
%   rate FS, and optional arguments NUMTARGETS and TEMPNUM (taken to be =
%   #songs, 1 if not passed in).
%   
%   For example, you can calculate the d' scores for a standard spikeMatrix
%   as:
%  
%    taus=[0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1];
%    plot(taus,calcdp(spikeMatrix(:,:,:,1),taus,1000))
%

if(nargin < 7 || isempty(useTrials))
    useTrials = false;
end
if(nargin < 6 || isempty(useMins))
    useMins = false;
end
[numTrials,numSongs] = size(st);
if(nargin < 5 || isempty(fs))
    fs = 10000;
end
if(nargin < 4 || isempty(tempNum))
    tempNum = 1;
end
dur = max(max(vector(cellfun(@sum,cellfun(@max,st,'UniformOutput',false)))),0);
sm = st2sm(st,fs,dur);
Ns = size(sm,1);
numTaus = length(taus);

numVars = numSongs/numTargets;
% dp will store the d' scores for each tau
nJN = 1;
numTrialsJN = numTrials;
if(numTrials >= 3) % Need at least 3 trials to jacknife
    nJN = numTrials;
    numTrialsJN = numTrials-1;
end
vinds = triu(true(numTargets-1,numTargets),1);

if(useTrials)
	testTrialInds = vector(toeplitz(1:numTrialsJN-1,[1 numTrialsJN:-1:2]));
    templateTrialInds = reshape(repmat(1:numTrialsJN,numTrialsJN-1,1),numTrialsJN*(numTrialsJN-1),1);
else
	testTrialInds = 1:numTrialsJN;
    templateTrialInds = testTrialInds;
end
nTotal = numTrials*numSongs;

dp = zeros(numTaus,numVars,nJN);
for ti = 1:numTaus
    % Convolve with a normal kernel of the appropriate size
	padLen = ceil(taus(ti)*fs)*3; % Zero pad 3x the stdev
	pad = zeros(padLen,numTrials*numSongs);
    smConv = reshape(normconv([pad;sm(:,:);pad],taus(ti)*fs),Ns+2*padLen,numTrials,numSongs);

	% Compute "distance matrix"
    if(useTrials)
        dists = smConv(:,:).'*smConv(:,:);
    else
        dists = bsxfun(@minus,sum(smConv,2),smConv);
        dists = smConv(:,:).'*dists(:,:);
    end
    dinds = reshape(1:nTotal,numTrials,numSongs);

    for ji = 1:nJN
        if(nJN > 1)
            indsJN = [1:ji-1 ji+1:numTrials];
        else 
            indsJN = 1:numTrials;
        end
        templateSongInds = (0:numTargets-1)*numVars+tempNum;
        dpTemp = zeros(numVars,numTargets-1,numTargets);
        % Calculate the d' between each pair of stimuli
        for vi = 1:numVars
            testSongInds = (0:numTargets-1)*numVars+vi;
            for ai = 1:numTargets-1
                for bi = ai+1:numTargets
                    indat = dinds(indsJN(templateTrialInds),templateSongInds(ai));
                    indbt = dinds(indsJN(templateTrialInds),templateSongInds(bi));
                    inda = dinds(indsJN(testTrialInds),testSongInds(ai));
                    indb = dinds(indsJN(testTrialInds),testSongInds(bi));
                    a = dists(inda+(indat-1)*nTotal) - dists(inda+(indbt-1)*nTotal);
                    b = dists(indb+(indat-1)*nTotal) - dists(indb+(indbt-1)*nTotal);

                    % Keep a running tally of d' scores
                    dpTemp(vi,ai,bi) = fastdp(a,b);
                    dpTemp(vi,bi-1,ai) = dpTemp(vi,ai,bi);
                end
            end
        end
        if(useMins)
            dp(ti,:,ji) = mean(min(dpTemp,[],2),3);
        else
            dp(ti,:,ji) = mean(dpTemp(:,vinds),2);
        end
    end
end
% Normalize by the right factor
dp = dp*sqrt(2);
stdJN = std(dp,[],3);
dp = mean(dp,3);

function dp=fastdp(a,b)
% dp = abs(mean(a)-mean(b))/sqrt(var(a)+var(b));
a = [a b];
m = mean(a);
denom = max(size(a,1)-1,1);
v = sum(bsxfun(@minus,a,m).^2)/denom;
dp=(m(1)-m(2))/sqrt(v(1)+v(2));
