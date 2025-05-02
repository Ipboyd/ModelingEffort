%close all;
function plotAll_digraphs(i,j, TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap)
source_offsetY = 4;
masker_offsetY = 4;

source_labels = [0,1,2,3,4];
masker_labels = [0,1,2,3,4];

pre_normalized_x = [0.59, 1.24, 1.91, 2.57];

% initialize
x_normalized = [0,0];
y_normalized = [0,0];


% Get the limits of the plot
%ax = gca;
x_range = [0.75,5.25];
y_range = [-1.5,4];

weights = ones(1,length(targets))*1;

G = digraph(sources,targets,weights);
 

% Define the desired size of the figure in pixels
figureWidth = 1200; % Width in pixels
figureHeight = 900; % Height in pixels

tic;
% for i=source_labels
%     for j=masker_labels
        % everything but s0m0
        if i == 0 && j == 0
            % continue;
        end
        source_offsetX = 0.25 + (i-1); 
        masker_offsetX = 0.25 + (j-1);
        
        % Create the figure with the specified size
        figure('Position', [100, 100, figureWidth, figureHeight]);
        
        % graph and change X and PV attributes
        p = plot(G,'k','Xdata',Nodesx,'Ydata',Nodesy); hold on

        axis([0 5.25 -1 4]);
        
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
        if i~=0 && j~=0
            if i == j
                xx = pre_normalized_x(i) + 1;
                create_arrow(xx, 'Source + Masker', 'blue', x_range, y_range);
            else
                if i ~= 0
                    xx = pre_normalized_x(i) + 1;
                    create_arrow(xx, 'Source', 'green', x_range, y_range);
                end
                if j ~= 0
                    xx = pre_normalized_x(j) + 1;
                    create_arrow(xx, 'Masker', 'red', x_range, y_range);
                end
            end
        end
        

        % Set up the ButtonDownFcn for the nodes
        set(p, 'ButtonDownFcn', @(src, event) node_click_callback(src, event, i, j, Nodesx, Nodesy, reverseNodeMap));

%     end
% end
toc;
       

% Callback function to handle node clicks
function node_click_callback(src, event, source_label, masker_label, Nodesx, Nodesy, reverseNodeMap)
    % Get the clicked node
    % Get the click coordinates in the axes
    clickCoords = get(gca, 'CurrentPoint');
    clickX = clickCoords(1,1);
    clickY = clickCoords(1,2);
    
    % Calculate distances to all nodes
    distances = sqrt((Nodesx - clickX).^2 + (Nodesy - clickY).^2);
    
    % Find the nearest node
    [~, nodeIndex] = min(distances);
    
    % Construct the file names for the plots
    nodeType = determine_node_type(nodeIndex);
    nodeClass = reverseNodeMap(nodeType); % finds node class for file naming

    % use index to figure out what channel the node is in
    quotient = floor(nodeIndex/7);

    % not efficient yet for C, since theres only 1 C for all channels
    % cc is the channel of the node clicked
    if nodeIndex == 29 || nodeType == 7 % cortical or X
        cc = quotient;
    else
        cc = quotient + 1;
    end

    rasterPlotFile = sprintf('Plot_Structure_Graphs/%s%d_s%dm%d_RASTER.fig', nodeClass, cc, source_label, masker_label);
    psthPlotFile = sprintf('Plot_Structure_Graphs/%s%d_s%dm%d_PSTH.fig', nodeClass, cc, source_label, masker_label);
    
    % Display the plots
    if isfile(rasterPlotFile)
        openfig(rasterPlotFile, 'new', 'visible');
    else
        disp('Raster plot file not found.');
    end
    
    if isfile(psthPlotFile)
        openfig(psthPlotFile, 'new', 'visible');
    else
        disp('PSTH plot file not found.');
    end
end
end

% determine node type based on node index
function nodeType = determine_node_type(nodeIndex)
    if nodeIndex == 29
        nodeType = 29;
    else
        nodeType = mod(nodeIndex, 7);
        if nodeType == 0
            nodeType = 7;
        end
    end
end


% Define the function for creating textarrow annotations
function create_arrow(xx, arrow_string, arrow_color, x_range, y_range)
    % Convert data coordinates to normalized figure units
    x_normalized = ([xx, xx] - x_range(1)) / (x_range(2) - x_range(1)) + 0.13;
    y_normalized = ([-0.55, -0.15] - y_range(1)) / (y_range(2) - y_range(1));
    
    % Draw the annotation
    annotation('textarrow', x_normalized, y_normalized, 'String', arrow_string, 'FontSize', 12, 'Color', arrow_color, 'LineWidth', 5);
end


