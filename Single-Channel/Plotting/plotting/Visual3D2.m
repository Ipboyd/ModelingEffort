%Idea for future implementation

%1. Do a 3D grid search
%2. Create a function that converts the parpameters into a matrix of
%verticies.
%3. Create a function that colormaps the resulting things.

x_len = 5;
y_len = 5;
z_len = 5;
gap = 0.25;


x_vec = linspace(0,(x_len-1)+(gap*x_len),x_len);
y_vec = linspace(0,(y_len-1)+(gap*y_len),y_len);
z_vec = linspace(0,(z_len-1)+(gap*z_len),z_len);

% Initialize an empty cell array to store permutations
perms = {};

% Nested loops to generate permutations
for i = 1:length(x_vec)
    for j = 1:length(y_vec)
        for k = 1:length(z_vec)
            perms{end+1} = [x_vec(i), y_vec(j), z_vec(k)];
        end
    end
end

% Convert cell array to matrix
perms = cell2mat(perms');

% Display permutations
%disp(perms);

vertex1_coordinates = perms;

%Playing with alphas
alphas = (pc(15).perf.SPIKE./pc(15).perf.ISI -1) + (pc(15).perf.RISPIKE./pc(15).perf.ISI -1);


normYpred = alphas - min(alphas);
normYpred = (normYpred ./ max(normYpred));

%normYpred(normYpred<0.5) = 0;
normYpred(normYpred<0.7) = 0;

%normYpred = (normYpred).^4;

%%%Plot firing rate

number_list = pc(15).fr;
plotCubes(vertex1_coordinates, number_list,normYpred);

xticklabels = [0,0.11,0.115,0.117,0.12];
xticks = x_vec+0.5;
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = [0,0.01,0.011,0.012,0.013];
yticks = y_vec+0.5;
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
zticklabels = [0,0.013,0.015,0.016,0.017];
zticks = z_vec+0.5;
set(gca, 'ZTick', zticks, 'ZTickLabel', zticklabels)
title('Firing Rate')

%%%Plot Scoring

number_list = (pc(15).perf.SPIKE./pc(15).perf.ISI -1) + (pc(15).perf.RISPIKE./pc(15).perf.ISI -1);
plotCubes(vertex1_coordinates, number_list,normYpred);

xticklabels = [0,0.11,0.115,0.117,0.12];
xticks = x_vec+0.5;
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = [0,0.01,0.011,0.012,0.013];
yticks = y_vec+0.5;
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
zticklabels = [0,0.013,0.015,0.016,0.017];
zticks = z_vec+0.5;
set(gca, 'ZTick', zticks, 'ZTickLabel', zticklabels)
title('Score')

%%%Plot Max Score

number_list = max([pc(15).perf.SPIKE;pc(15).perf.ISI;pc(15).perf.RISPIKE]);
plotCubes(vertex1_coordinates, number_list,normYpred);

xticklabels = [0,0.11,0.115,0.117,0.12];
xticks = x_vec+0.5;
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = [0,0.01,0.011,0.012,0.013];
yticks = y_vec+0.5;
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
zticklabels = [0,0.013,0.015,0.016,0.017];
zticks = z_vec+0.5;
set(gca, 'ZTick', zticks, 'ZTickLabel', zticklabels)
title('Max')



function plotCubes(vertex1_coordinates, number_list,normYpred)
    % Define cube colors based on number_list
    colormap_name = 'parula'; % You can choose any colormap you like
    % mymap = [0 0 0
    % 1 1 1];
    colormap_vals = colormap(colormap_name);
    num_colors = size(colormap_vals, 1);
    min_val = min(number_list);
    max_val = max(number_list);
    color_indices = round(interp1(linspace(min_val, max_val, num_colors), 1:num_colors, number_list, 'linear', 'extrap'));

    % Initialize figure
    figure;
    hold on;

    % Define cube vertices and faces
    vertices = [0 0 0;   % Vertex 1
                1 0 0;   % Vertex 2
                1 1 0;   % Vertex 3
                0 1 0;   % Vertex 4
                0 0 1;   % Vertex 5
                1 0 1;   % Vertex 6
                1 1 1;   % Vertex 7
                0 1 1];  % Vertex 8

    faces = [1 2 3 4;    % Bottom face
             5 6 7 8;    % Top face
             1 2 6 5;    % Side face 1
             2 3 7 6;    % Side face 2
             3 4 8 7;    % Side face 3
             4 1 5 8];   % Side face 4

    % Plot each cube with corresponding color
    for i = 1:size(vertex1_coordinates, 1)
        % Get color based on number_list
        color_index = color_indices(i);
        if color_index < 1
            color_index = 1;
        elseif color_index > num_colors
            color_index = num_colors;
        end
        color = colormap_vals(color_index, :);
        
        % Translate vertices based on Vertex 1 coordinates
        translated_vertices = bsxfun(@plus, vertices, vertex1_coordinates(i, :));
        
        % Plot cube with specified color
        patch('Vertices', translated_vertices, 'Faces', faces, 'FaceColor', color, 'FaceAlpha', normYpred(i));
    end

    % Set plot properties
    xlabel('Gain');
    ylabel('On->PV strength');
    zlabel('Off->PV Strength');
    axis equal;
    grid on;
    view(3); % 3D view
end