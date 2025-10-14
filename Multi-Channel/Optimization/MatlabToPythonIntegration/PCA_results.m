addpath('results\')

%PSTH + VR 100ms
%m = matfile('run_2025-09-11_20-28-42.mat');

%PSTH only
%m = matfile('run_2025-09-05_22-19-39.mat');

%m = matfile('run_2025-09-24_12-23-46.mat');

%Layer 4
%m = matfile('run_2025-10-01_12-07-07.mat');

%m = matfile('run_2025-10-01_12-20-18.mat');

m = matfile('run_2025-10-01_12-42-12.mat');

losses = m.losses;
param_tracker = m.param_tracker;
output = m.output;


pc_axis1 = 1;
pc_axis2 = 2;

% A: [E × P × B]
[E, P, B] = size(param_tracker);

% Reorder to [E × B × P], then flatten to [E*B × P]
samples = reshape(permute(param_tracker, [1 3 2]), [], P);   % (E*B) × P

% Flatten loss to match rows in samples

loss = squeeze(losses(:,2,:));
loss_flat = loss(:);      

[coeff, score, latent, tsquared, explained, mu] = pca(samples, ...
    'Algorithm','svd', 'NumComponents', 10, 'Rows','complete');


figure;
scatter(score(:,pc_axis1), score(:,pc_axis2), 15, loss_flat, 'filled'); % size 15 marker
colorbar; ylabel(colorbar, 'Loss');
xlabel(sprintf('pc_axis1 (%.1f%% var)', explained(pc_axis1)));
ylabel(sprintf('pc_axis2 (%.1f%% var)', explained(pc_axis2)));
title('PCA of Parameters across Epochs × Batches (colored by Loss)');
axis equal; grid on;
hold on;

%Plot the convergent values
f_index = 400:5:2000;
f_scores = score(f_index,:);

plot(f_scores(:,pc_axis1), f_scores(:,pc_axis2), 'go', 'LineWidth', 1.5, 'MarkerSize', 4);

hold on;

% Reshape scores back to [E × B × 2]
score_eb2 = reshape(score(:,([pc_axis1,pc_axis2])), [E, B, 2]);

pc1_mean = squeeze(mean(score_eb2(:,:,1), 2));  % [E × 1]
pc2_mean = squeeze(mean(score_eb2(:,:,2), 2));  % [E × 1]


plot(pc1_mean, pc2_mean, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
legend('Samples (colored by loss)', 'Final Convergent value for each trial','Epoch mean trajectory');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% score: (E*B)×2, loss_flat: (E*B)×1

% Cast to double (and pull off GPU if needed)
x = double(gather(score(:,pc_axis1)));
y = double(gather(score(:,pc_axis2)));
z = double(gather(loss_flat(:)));

% Remove any NaN/Inf rows
ok = isfinite(x) & isfinite(y) & isfinite(z);
x = x(ok); y = y(ok); z = z(ok);

% Grid over PC space
nx = 300; ny = 300;
xlims = [min(x) max(x)];
ylims = [min(y) max(y)];
[xq,yq] = meshgrid(linspace(xlims(1),xlims(2),nx), ...
                   linspace(ylims(1),ylims(2),ny));

% Nearest-loss background
F = scatteredInterpolant(x, y, z,'natural','none');
Z = F(xq, yq);

% Mask outside convex hull (optional)
DT = delaunayTriangulation(x, y);
inside = ~isnan(pointLocation(DT, xq(:), yq(:)));
Z(~reshape(inside, size(Z))) = NaN;

% % Plot
% figure;
% imagesc(xlims, ylims, Z); set(gca,'YDir','normal'); axis equal tight; hold on;
% colormap(parula); cb=colorbar; ylabel(cb,'Loss');
% 
% %scatter(x, y, 8, z, 'filled', 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.85);
% xlabel('PC1'); ylabel('PC2');
% title('PC space colored by nearest-loss field');

% --- after you compute Z and mask with NaNs ---
figure;
himg = imagesc(xlims, ylims, Z);
set(himg,'AlphaData', isfinite(Z));     % transparent where Z is NaN
set(gca,'YDir','normal','Color','w');   % black background shows through
axis equal tight; hold on;
colormap(turbo); cb = colorbar; ylabel(cb,'Loss');
xlabel('PC1'); ylabel('PC2'); title('PC space colored by nearest-loss field');

hold on;
plot(f_scores(:,pc_axis1), f_scores(:,pc_axis2), 'go', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;

K = convexHull(DT);                    % hull vertex order (counter-clockwise)
plot(DT.Points(K,1), DT.Points(K,2), 'w-', 'LineWidth', 0.5);   % black edge on top




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Visualize parameters with aheat map?
% % Example: 20 trials, 10 parameters
% nTrials = 50; %nEpochs
% nParams = 10;
% %X = randn(nTrials, nParams);  % replace with your solutions
% 
% % Optional: normalize each parameter (z-score across trials)
% %Xz = (X - mean(X,1)) ./ std(X,[],1);
% 
% % Parameter and trial labels
% paramNames = arrayfun(@(i) sprintf("p%d", i), 1:nParams, 'UniformOutput', false);
% trialLabels = arrayfun(@(i) sprintf("trial %d", i), 1:nTrials, 'UniformOutput', false);
% 
% % Make the heatmap
% figure;
% 
% %SORTING BY FIRST COLUMN (On-R1On)
% h = heatmap(transpose(squeeze(param_tracker(nTrials,1:7,:))));
% %h = heatmap(transpose(squeeze(param_tracker(300,:,:))));
% % Customize appearance
% h.Colormap = parula;
% h.ColorbarVisible = 'on';
% h.XLabel = 'Parameters';
% h.YLabel = 'Trials';
% h.Title = 'Parameter Solutions Heatmap';
% 
% 
% % Make the heatmap
% figure;
% 
% %SORTING BY FIRST COLUMN (On-R1On)
% h = heatmap(transpose(squeeze(param_tracker(nTrials,8:2:26,:))));
% %h = heatmap(transpose(squeeze(param_tracker(300,:,:))));
% % Customize appearance
% h.Colormap = parula;
% h.ColorbarVisible = 'on';
% h.XLabel = 'Parameters';
% h.YLabel = 'Trials';
% h.Title = 'Parameter Solutions Heatmap';
% 
% % Make the heatmap
% figure;
% 
% %SORTING BY FIRST COLUMN (On-R1On)
% h = heatmap(transpose(squeeze(param_tracker(nTrials,9:2:27,:))));
% %h = heatmap(transpose(squeeze(param_tracker(300,:,:))));
% % Customize appearance
% h.Colormap = parula;
% h.ColorbarVisible = 'on';
% h.XLabel = 'Parameters';
% h.YLabel = 'Trials';
% h.Title = 'Parameter Solutions Heatmap';
% 
% 
% %SORTING BY FIRST COLUMN (On-R1On)
% h = heatmap(transpose(squeeze(param_tracker(nTrials,28,:))));
% %h = heatmap(transpose(squeeze(param_tracker(300,:,:))));
% % Customize appearance
% h.Colormap = parula;
% h.ColorbarVisible = 'on';
% h.XLabel = 'Parameters';
% h.YLabel = 'Trials';
% h.Title = 'Parameter Solutions Heatmap';


