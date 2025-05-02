

function persistent_gui(TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap)
    % Create the main figure
    structure_data = load('curr_data_structure.mat');
    fig = figure('Position', [100, 100, 400, 300], 'MenuBar', 'none', 'Name', 'Select Source and Masker', 'NumberTitle', 'off', 'Resize', 'off');
    
    % Create the text labels
    uicontrol('Style', 'text', 'Position', [50, 240, 50, 20], 'String', 'Source:');
    uicontrol('Style', 'text', 'Position', [50, 160, 50, 20], 'String', 'Target:');
    
    % Create buttons for selecting values of i
    i_buttons = gobjects(1, 5);
    for k = 0:4
        i_buttons(k + 1) = uicontrol('Style', 'pushbutton', 'Position', [100 + k*40, 240, 30, 30], 'String', num2str(k), 'Callback', @(src, event) select_i(k));
    end
    
    % Create buttons for selecting values of j
    j_buttons = gobjects(1, 5);
    for k = 0:4
        j_buttons(k + 1) = uicontrol('Style', 'pushbutton', 'Position', [100 + k*40, 160, 30, 30], 'String', num2str(k), 'Callback', @(src, event) select_j(k));
    end
    
    % Create the button to confirm selection and plot
    uicontrol('Style', 'pushbutton', 'Position', [150, 50, 100, 30], 'String', 'Plot', 'Callback', @plot_callback);
    
    % Initialize variables for i and j
    selected_i = 0;
    selected_j = 0;

    % Callback function for selecting i
    function select_i(val)
        selected_i = val;
        disp(['Selected i: ', num2str(selected_i)]);
        % Reset the background color of all i buttons
        for k = 1:5
            set(i_buttons(k), 'BackgroundColor', [0.94, 0.94, 0.94]); % Default background color
        end
        % Highlight the selected i button
        set(i_buttons(val + 1), 'BackgroundColor', [0.7, 0.7, 0.7]); % Highlight color
    end

    % Callback function for selecting j
    function select_j(val)
        selected_j = val;
        disp(['Selected j: ', num2str(selected_j)]);
        % Reset the background color of all j buttons
        for k = 1:5
            set(j_buttons(k), 'BackgroundColor', [0.94, 0.94, 0.94]); % Default background color
        end
        % Highlight the selected j button
        set(j_buttons(val + 1), 'BackgroundColor', [0.7, 0.7, 0.7]); % Highlight color
    end

    % Callback function for plot button
    function plot_callback(~, ~)
        % Call the main plotting function with selected i and j
        main_function(selected_i, selected_j, TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap);
    end
end

function main_function(i, j, TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap)
    plotAll_digraphs(i, j, TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap);
    disp([i, j]);
end