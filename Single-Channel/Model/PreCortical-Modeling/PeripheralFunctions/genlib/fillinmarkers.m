function fillinmarkers(h, recolor)
% fillinmarkers(h, recolor)
%
% Fills in all the markers in all the input handles and their progeny.  No
% argument will result in every line object in the current figure to be filled
% in.  An argument of 0 will fill in all markers in all figures.
%
% If the second argument is nonzero, markers whose colors have already been
% specified (i.e., MarkerFaceColor not set to 'none') will be recolored
% according to their line's color.  This defaults to 0, which means the already-
% specified markers stay the same color.


if nargin < 1
	h = gcf;
end
if nargin < 2
	recolor = 0;
end

%% new version
if recolor
	for ii = 1:length(h)
		traverseGraphicsTree(h(ii), 'Line', {'MarkerFaceColor', '%get(hObject, ''Color'')'})
	end
else
	for ii = 1:length(h)
		traverseGraphicsTree(h(ii), '%strcmpi(get(hObject, ''Type''), ''Line'') && strcmpi(get(hObject, ''MarkerFaceColor''), ''None'')',...
			{'MarkerFaceColor', '%get(hObject, ''Color'')'})
	end
end

%% old version
% h_all = [];
% 
% if h == 0
% 	h_figs = get(0, 'children');
% 	for ii = 1:length(h_figs)
% 		h_all = [h_all; [h_figs(ii); get(h_figs(ii), 'children'); get(get(h_figs(ii), 'children'), 'children')]];
% 	end
% else
% 	h_all = [h; get(h, 'children'); get(get(h, 'children'), 'children')];
% end
% 
% if ~iscell(h_all)
% 	h_all = {h_all};
% end
% for ii = 1:length(h_all)
% 	for jj = 1:length(h_all{ii})
% 		if strcmpi(get(h_all{ii}(jj), 'type'), 'line')
% 			if strcmpi(get(h_all{ii}(jj), 'markerfacecolor'), 'none') || recolor
% 				set(h_all{ii}(jj), 'markerfacecolor', get(h_all{ii}(jj), 'color'))
% 			end
% 		end
% 	end
end