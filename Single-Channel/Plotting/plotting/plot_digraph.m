%% Graphing Segment

%Be careful! Do not override the s struct

weights = ones(1,length(targets))*1;

G = digraph(sources,targets,weights);

% Create the figure with the specified size
figure('Position', [100, 100, 800, 600]);

% graph and change X and PV attributes
p = plot(G,'k','Xdata',Nodesx,'Ydata',Nodesy); hold on

% highlight P-E connections
highlight(p, inhibs_PEs,'NodeColor','red')
for node=1:length(inhibs_PEt)
    if inhibs_PEt{node}{2} == 1 % within channel
        highlight(p, inhibs_PEs(node), inhibs_PEt{node}{1},'EdgeColor','blue')
    else                        % cross channel
        highlight(p, inhibs_PEs(node), inhibs_PEt{node}{1},'EdgeColor','green')
    end
end

% highlight X-R connections
highlight(p, SOM_nodes,'NodeColor','red')
highlight(p, inhibs_XRs, inhibs_XRt,'EdgeColor','red')
%highlight(p, inhibs_XRs, inhibs_XRt,'LineStyle','--')

% highlight TD-X connections
highlight(p, TD_SOMs,'NodeColor','red')
highlight(p, TD_SOMs, TD_SOMt,'EdgeColor',[0.75,0.75,0])

% highlight and modify line widths and arrow sizes
for node=1:length(sources)
    highlight(p, sources(node), targets(node), 'LineWidth',all_gsyns(node),'ArrowSize',arrow_sizes(node))
end

% % Get the limits of the plot
% ax = gca;
% xlim = [.25,4.75];
% ylim = ax.YLim;
% 
% % pre_normalized_x = [0.59, 1.24, 1.91, 2.57];
% 
% 
% xx = 2.57 + 1;
% % Convert data coordinates to normalized figure units
% x_normalized = ([xx, xx] - xlim(1)) / (xlim(2) - xlim(1));
% y_normalized = ([3.5,2.75] - ylim(1)) / (ylim(2) - ylim(1));
% 
% annotation('textarrow', x_normalized, y_normalized, 'String', 'Source', 'FontSize', 12, 'Color', 'blue');