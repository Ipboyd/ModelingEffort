function varargout = rasterplot(in, fs, color, offset)
% rasterplot(spikemat/spiketimes, fs, color, offset)
% last two are optional and default to black and 0 respectively.

if nargin == 1
	fs = 1;
end

if nargin <= 2
	color = 'k';
end

if nargin <= 3
	offset = 0;
end

newplot

len = 0;
if ~iscell(in)
	len = size(in, 1)/fs;
	x = sm2st(in, fs);
else
	x = in;
end
[numtrials, numsongs] = size(x);
h = zeros(numtrials, numsongs); % a matrix of line object handles for output purposes
if(nargout == 1)
    varargout{1} = h;
end

if ~sumall(cellfun(@length, x))
	warning('MATLAB:noSpikes', 'There were no spikes, so RASTERPLOT did nothing.')
	return
end

for song = 1:numsongs
	for trial = 1:numtrials
		num = (song - 1)*numtrials + trial;
        numspikes = length(x{trial, song});
        xx = x{trial, song}(:).' + offset;
		yy = ones(size(x{trial, song})).'*num - 0.5;
        xx = reshape([xx;xx;nan(1,numspikes)],numspikes*3,1);
        yy = reshape([yy+0.1;yy+1-0.1;nan(1,numspikes)],numspikes*3,1);

        if ~isempty(xx)
			h(trial, song) = line(xx, yy, 'linestyle', '-', 'linewidth', 0.5, 'color', color);
		end
		if ~isempty(x{trial, song})
			len = max(len, x{trial, song}(end));
		end
	end
end
axis([offset, len + offset, 0.5, num + 0.5])
set(gca, 'YDir', 'Reverse')
if(strcmpi(get(gca, 'YTickMode'), 'auto'))
	set(gca, 'YTick', [numtrials + 0.5:numtrials:numsongs*numtrials], 'YTickLabel', [], 'YGrid', 'On', 'GridLineStyle', '-')
end

if nargout == 1
	varargout{1} = h;
end