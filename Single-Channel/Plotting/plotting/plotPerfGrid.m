function plotPerfGrid(neuronPerf, neuronFR, titleString, textColorThresh)
    % plotPerfGrid(neuronPerf, neuronFR, titleString, textColorThresh)
    %
    % plot single performance grids; can handle a data array of 16 or 8 elements
    % works with mouseNetwork_main
    if ~exist('textColorThresh','var'), textColorThresh = 70; end
    if ~exist('neuronFR','var'), neuronFR = 'n/a'; end

    % Determine the grid size and reshape performance data
    if numel(neuronPerf) == 16 % mixed grid
        [X, Y] = meshgrid(1:4, 4:-1:1);
        neuronPerf = reshape(neuronPerf, 4, 4);
    elseif numel(neuronPerf) == 8 % target or masker only
        [X, Y] = meshgrid(1:4, 1:2);
    elseif numel(neuronPerf) == 4 % target only
        [X, Y] = meshgrid(1:4, 1);
    end

    % Convert performance data to string
    str = cellstr(num2str(round(neuronPerf(:))));
    str2 = cellstr(num2str(round(neuronFR(:))));

    % Display the grid using imagesc
    hImg = imagesc(flipud(neuronPerf));
    set(hImg, 'ButtonDownFcn', @imageClickCallback);  % Set the click callback

    % Customize axes and labels
    if numel(neuronPerf) == 16
        xticks([1:4]); xticklabels([]);
        yticks([1:4]); yticklabels([]);
    elseif numel(neuronPerf) == 8
        xticks([]); yticks([]);
    elseif numel(neuronPerf) == 4
        xticks([]); yticks([]);
    end

    % Add text labels
    t = text(X(:), Y(:) - 0.15, str, 'Fontsize', 12, 'HorizontalAlignment', 'Center');
    t2 = text(X(:), Y(:) + 0.15, strcat('(', str2, ')'), 'Fontsize', 12, 'HorizontalAlignment', 'Center');

    % Set text colors based on performance values
    for i = 1:numel(neuronPerf)
        if neuronPerf(i) > textColorThresh
            t(i).Color = 'k';
            t2(i).Color = 'k';
        else
            t(i).Color = 'w';
            t2(i).Color = 'w';
        end
    end

    % Customize appearance
    title(titleString);
    colormap('parula');
    caxis([50, 90]);
    set(gca, 'fontsize', 12);
    set(gcf, 'MenuBar', 'none', 'Toolbar', 'none');

    % Click callback function for the image
    function imageClickCallback(src, event)
        % Get the coordinates of the click
        coords = get(gca, 'CurrentPoint');
        x = coords(1, 1);
        y = coords(1, 2);

        % Determine the clicked cell
        i = ceil(x);
        j = size(neuronPerf, 1) - floor(y) + 1;

        % Highlight the clicked cell
        hold on;
        plot(i, j, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;

        % Display clicked coordinates
        disp(['Clicked: Cell (' num2str(j) ', ' num2str(i) ')']);
    end
end
