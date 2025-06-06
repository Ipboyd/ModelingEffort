function hh = tightenticklabel(ax)
%
% TWXTICKLABEL tweaks the xtick labels of the current axis, so that fonts
% are presereved when the figure is exported to eps (e.g. for inclusion
% into LaTeX).
%
% TWXTICKLABEL(AX) applies TWXTICKLABEL to the axis with handle AX.
%
% H = TWXTICKLABEL  returns in H the handles of all XTick labels so the
% user can change every property which is possible with a text object created by text command.
% Following are a few examples:
%
% h = twxticklabel;
% set(h,'fontsize',12,'fontname','courier'); %  to change font namae and size
% set(h,{'string'},char('10','-3','pi','he','90','ju')); % to change ticklabels
% set(h,'rotation',90); % to change angle
%
% The two m-files TWXTICKLABEL and TWYTICKLABEL motivated by the following problems in the way Matlab handles tick labels.
%1)
% Matlab provides no handles to xtick labels and ytick labels; meaning that you can't change their
% properties separately, but only by changing axis properties. For example, the only way to change
% the fontname of xtick labels is by setting the fontname property of the axis, thus changing the fontname
% of all it object, which may not be desirable. If a user wants to rotate the tick labels, one would
% like to be able to do that with commands like set(handle, 'rotation', 90); but that is only possible if you have
% the handle.
%
%2)
% Tick labels do not interpret TeX/LaTeX character sequences; What if you want LaTeX expression $W_i(t)$.
%
%3)
% Even if you are happy with changing the properties of the whole axis, another problem arises when the Matlab
% figure is exported to eps for inclusion into LaTeX. It sometimes changes the fonts (set by the user) of the
% xticklabels in an unpredictable way (fortunately, this does not happen with xlabel, ylabel, title and other text object).
%
%
% In these two files, I have tried to create ticklabels by using text command. Handles are returned for all tick labels,
% so the user can change every property which is possible with a text object created by text command.

if nargin == 0
	ax = gca;
end
hh = [];
for ii = 1:length(ax)
	hh = [hh; dothework(ax(ii))];
end
end


function hh = dothework(ax)
xtick = get(ax,'XTick')';

if isempty(xtick)
	warning('''XTick'' found empty')
else
	list = 'x'; % make this an argument later
	if ~isempty(find(lower(list), 'x'))
		hxlabel = get(ax,'xlabel');
		oldUnits = get(hxlabel, 'Units');
		set(hxlabel,'Units','data');
		xticklab = get(ax,'XTickLabel');
		if ~iscell(xticklab)
			xticklab = strjust(xticklab, 'center'); % RKM do not delete trailing or preceding spaces
			xticklab = strtrim(mat2cell(xticklab, ones(1, size(xticklab, 1)), size(xticklab, 2)));
		end
		set(ax,'XTickLabel',[])
		
		ylims = get(ax,'ylim');
		if strcmpi(get(ax, 'YDir'), 'normal')
			xtickloc = ylims(1);
		else
			xtickloc = ylims(2);
		end
		if length(xticklab) < length(xtick)
			xtick = xtick(1:length(xticklab));
		end
		
		axes(ax)
		h = text(xtick, xtickloc(ones(size(xtick))), xticklab);
		
		set(h, 'fontsize', get(ax,'fontsize'), ...
			'HorizontalAlignment', 'center', 'VerticalAlignment', 'top')
		
		set(hxlabel,'Units',oldUnits);
	end
end

if nargout == 1 && exist('h', 'var')
	hh = h;
else
	hh = NaN;
end
end