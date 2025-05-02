% Example code for plotting voxels
% Generate some sample voxel data
voxel_data = randi([0, 1], [10, 10, 10]); % Binary voxel data (0 or 1)

% Define the grid dimensions
[x, y, z] = meshgrid(1:size(voxel_data, 1), 1:size(voxel_data, 2), 1:size(voxel_data, 3));

% Extract the voxel centers for voxels with value 1
[x_voxel, y_voxel, z_voxel] = ind2sub(size(voxel_data), find(voxel_data));

% Plot the voxels
figure;
hold on;
for i = 1:length(x_voxel)
    % Draw a small cube (voxel) at each voxel center
    patch(x_voxel(i) + [-0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5], ...
          y_voxel(i) + [-0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5], ...
          z_voxel(i) + [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5], ...
          'b', 'FaceAlpha', 0.5);
end

% Set plot properties
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal;
grid on;
view(3); % 3D view