function varargout = psthplot(in, fs, color, binsize, offset)
% rasterplot(spikemat/spiketimes, fs, color, offset)
% last two are optional and default to black and 0 respectively.
%adapted from rasterplot.m 8/25/10

if nargin == 1
	fs = 1;
end

if nargin <= 2
	color = 'k';
end

if nargin <= 3
	binsize = .02;
end

if nargin <= 4
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
if ~sumall(cellfun(@length, x))
	warning('MATLAB:noSpikes', 'There were no spikes, so PSTHPLOT did nothing.')
	return
end
[numtrials, numsongs] = size(x);

h = zeros(numtrials, numsongs); % a matrix of line object handles for output purposes

bins=cell(numsongs,1); edges=cell(numsongs,1);
for song = 1:numsongs
    spktms=x(:,song); spktms=vertcat(spktms{:});
    %bin in 10 ms bins
    edges{song,1}=0:binsize:ceil(max(spktms)*200)/200';
    bins{song,1}=histc(spktms,edges{song});
    %resize to fit between 0 and 10
    if ~isempty(spktms)
        len = max(len, max(spktms));
    end
end
maxcount=max(cellfun(@max,bins));
for song = 1:numsongs
    shift=10*(numsongs-song);
    h(song)=bar(edges{song},shift+(10*bins{song}/maxcount));
    set(h(song),'BaseValue',shift); hold on;
end
axis([offset, len + offset, 0, numsongs*10])
if(strcmpi(get(gca, 'YTickMode'), 'auto'))
	set(gca, 'YTick', [0:10:numsongs*10], 'YTickLabel', [], 'YGrid', 'On', 'GridLineStyle', '-')
end

if nargout == 1
	varargout{1} = h;
end