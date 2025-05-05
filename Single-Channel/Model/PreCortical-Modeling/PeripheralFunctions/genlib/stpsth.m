function varargout = stpsth(st, binTime, bounds)

if nargin < 2 || isempty(binTime)
	binTime = 20e-3;
end
if nargin < 3
	bounds = NaN;
end
[nTrials, nStimuli] = size(st);
maxTime = 0;
for trialNum = 1:nTrials
	for stimNum = 1:nStimuli
		maxTime = max(maxTime, sum(max(st{trialNum, stimNum})));
	end
end
bins = (0:binTime:maxTime + binTime).';
nBins = length(bins);
psth = zeros(nBins, nStimuli);
psthSTD = zeros(size(psth));
psthTrials = zeros(nBins, nTrials);
for stimNum = 1:nStimuli
	for trialNum = 1:nTrials
		if numel(st{trialNum, stimNum}) > 0
			psthTrials(:,trialNum) = hist(st{trialNum, stimNum}, bins);
		else
			psthTrials(:,trialNum) = zeros(nBins,1);
		end
	end
	psth(:,stimNum) = mean(psthTrials, 2);
	psthSTD(:,stimNum) = std(psthTrials, 1, 2);
end
psth = psth/binTime;
psthSTD = psthSTD/binTime;

if ~isnan(bounds)
	if bounds(1) < bins(1)
		bins = [flipud(bins(1):-binTime:bounds(1)); bins(2:end)];
		psth = [zeros(size(bins,1) - size(psth, 1), size(psth, 2)); psth];
		psthSTD = [zeros(size(bins,1) - size(psthSTD, 1), size(psthSTD, 2)); psthSTD];
	else
		bins = bins(bins >= bounds(1));
		psth = psth(end - size(bins, 1) + 1:end,:);
		psthSTD = psthSTD(end - size(bins, 1) + 1:end,:);
	end
	if bounds(2) > bins(end)
		bins = [bins(1:end - 1); bins(end):binTime:bounds(2)];
		psth = [psth, zeros(size(bins,1) - size(psth, 1), size(psth, 2))];
		psthSTD = [psth, zeros(size(bins,1) - size(psthSTD, 1), size(psthSTD, 2))];
	else
		bins = bins(bins <= bounds(2));
		psth = psth(1:size(bins, 1),:);
		psthSTD = psthSTD(1:size(bins, 1),:);
	end
end

stSize = size(st);
psth = reshape(psth, [size(psth, 1), stSize(2:end)]);
psthSTD = reshape(psthSTD, [size(psth, 1), stSize(2:end)]);

if nargout == 0
	imagesc(bins, 1:nStimuli, psth(:,:).')
	xlabel('Time (s)')
	ylabel('Stimulus')
 	set(gca, 'YTick', 1:nStimuli)
else
	varargout{1} = psth;
end
if nargout >= 2
	varargout{2} = bins;
end
if nargout >= 3
	varargout{3} = psthSTD;
end