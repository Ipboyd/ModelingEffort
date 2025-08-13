pc_axis1 = 9;
pc_axis2 = 10;

% A: [E × P × B]
[E, P, B] = size(param_tracker);

% Reorder to [E × B × P], then flatten to [E*B × P]
samples = reshape(permute(param_tracker, [1 3 2]), [], P);   % (E*B) × P

% Flatten loss to match rows in samples

loss = squeeze(losses(:,1,:));
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

% Reshape scores back to [E × B × 2]
score_eb2 = reshape(score(:,([pc_axis1,pc_axis2])), [E, B, 2]);

pc1_mean = squeeze(mean(score_eb2(:,:,1), 2));  % [E × 1]
pc2_mean = squeeze(mean(score_eb2(:,:,2), 2));  % [E × 1]

hold on;
plot(pc1_mean, pc2_mean, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
legend('Samples (colored by loss)', 'Epoch mean trajectory');

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
%DT = delaunayTriangulation(x, y);
%inside = ~isnan(pointLocation(DT, xq(:), yq(:)));
%Z(~reshape(inside, size(Z))) = NaN;

% Plot
figure;
imagesc(xlims, ylims, Z); set(gca,'YDir','normal'); axis equal tight; hold on;
colormap(turbo); cb=colorbar; ylabel(cb,'Loss');

scatter(x, y, 8, z, 'filled', 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.85);
xlabel('PC1'); ylabel('PC2');
title('PC space colored by nearest-loss field');