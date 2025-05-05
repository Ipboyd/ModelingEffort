function setabsticklength(h, tickLength, units, exempt0)
% setabsticklength(h, tickLength, units, exempt0)
%
% Sets the tick length in all the input handles and their progeny. An
% argument of 0 will affect all axes in all figures. Axes whose TickLength
% is already set to [0 0] will not be affected, unless the final argument
% is set to false (default is true--do exempt ticks with zero length).


if nargin == 1
	tickLength = h;
	h = gcf;
end
if nargin < 3
	units = 'inches';
end
if ~exist('exempt0', 'var')
	exempt0 = 1;
end

h_all = [];

if h == 0
	h_figs = get(0, 'children');
	for ii = 1:length(h_figs)
		h_all = [h_all; [h_figs(ii); get(h_figs(ii), 'children')]];
	end
else
	h_all = [h; get(h, 'children')];
end

if ~iscell(h_all)
	h_all = {h_all};
end
for ii = 1:size(h_all, 1)
	for jj = 1:size(h_all{ii}, 1)
		if strcmpi(get(h_all{ii}(jj), 'type'), 'axes')
			if ~isequal([0 0], get(h_all{ii}(jj), 'TickLen')) || ~exempt0
				oldUnits = get(h_all{ii}(jj), 'Units');
				set(h_all{ii}(jj), 'Units', units)
				position = get(h_all{ii}(jj), 'Position');
				set(h_all{ii}(jj), 'TickLength', tickLength.*[1 1]/max(position(3:4)))
				set(h_all{ii}(jj), 'Units', oldUnits)
			end
		end
	end
end