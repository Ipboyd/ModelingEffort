function t = sm2st(mat, fs)
% converts spikeMat(N,nTrials,nStimuli) to spikeTimes{nTrials,nStimuli}

matSize = size(mat);
outSize = matSize(2:end);
if length(outSize) == 1
	outSize = [outSize 1];
end
t = cell(outSize);
for ii = 1:prod(outSize)
	t{ii} = find(mat(:,ii))/fs;
end
t = reshape(t, outSize);