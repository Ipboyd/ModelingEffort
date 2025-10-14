%% ===== Config =====
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells');
addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Plotting\OliverDataPlotting');

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');

folder = 'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration\results_all_cells';
files  = dir(fullfile(folder,'*.mat'));
[~,idx] = sort([files.datenum]);
files = files(idx);                        % assumes this order matches all_data order

%% ===== Output directory for figures =====
outdir = fullfile(folder, 'plots_pv_gt_ee');
if ~exist(outdir, 'dir'), mkdir(outdir); end

pct_list = [0.1 0.5 1 2 5 10];            % percentiles of LOWEST loss to include (in %)
include_best_fit = true;                   % also evaluate per-cell best fit (min loss)
make_plots = true;                         % show scatterplots by percentile

param1 = 2;
param2 = 4;


layers = ["L2/3","L4","L5/6","NaN"];
cmap  = [ 57 106 177;
          62 150  81;
         204  37  41;
         160 160 160 ] / 255;

% Map text layer in all_data to index 1..4
layer_to_idx = @(s) find(layers == string(s), 1, 'first');

%% ===== Accumulators =====
% For each percentile k, store EE, PVE, and layer index
K = numel(pct_list);
acc(K).EE = [];  acc(K).PVE = [];  acc(K).L = []; %#ok<SAGROW>
if include_best_fit
    acc_best.EE = []; acc_best.PVE = []; acc_best.L = [];
end

%% ===== Collect points across cells =====
for k = 1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    S = load(fpath);   % expects fields: losses (epochs x 2 x runs), param_tracker (epochs x P x runs)
    % Loss matrix (epochs x runs), using column 2 as in your code
    losses = squeeze(S.losses(:,2,:));                 % [E x R]
    [E, R] = size(losses);

    % Layer index for this cell
    Li = layer_to_idx(all_data(k).layer);
    if isempty(Li), Li = 4; end                        % fallback to NaN group if unexpected text

    % Flatten helpers
    Lflat = losses(:);                                 % E*R x 1

    % ===== Best fit (per-cell minimum loss; allow ties) =====
    if include_best_fit
        mn = min(Lflat);
        idx_flat = find(Lflat == mn);                  % linear indices into [E x R]
        [row, col] = ind2sub([E R], idx_flat);
        % pull parameters: param_tracker(epoch, param_idx, run)
        EE_vals  = S.param_tracker(sub2ind([E R], row, col) + (1-1)*E*R); %#ok<NASGU>
        % More robustly, index per element:
        this_EE  = arrayfun(@(i) S.param_tracker(row(i), param1, col(i)), 1:numel(row))';
        this_PVE = arrayfun(@(i) S.param_tracker(row(i), param2, col(i)), 1:numel(row))';
        acc_best.EE  = [acc_best.EE;  this_EE];
        acc_best.PVE = [acc_best.PVE; this_PVE];
        acc_best.L   = [acc_best.L;   repmat(Li, numel(row), 1)];
    end

    % ===== Percentile selections (per cell) =====
    for kk = 1:K
        thr = prctile(Lflat, pct_list(kk));           % threshold at requested percentile
        idx_flat = find(Lflat <= thr);                 % all fits at/under the threshold
        if isempty(idx_flat), continue; end
        [row, col] = ind2sub([E R], idx_flat);

        this_EE  = arrayfun(@(i) S.param_tracker(row(i), param1, col(i)), 1:numel(row))';
        this_PVE = arrayfun(@(i) S.param_tracker(row(i), param2, col(i)), 1:numel(row))';

        acc(kk).EE  = [acc(kk).EE;  this_EE];
        acc(kk).PVE = [acc(kk).PVE; this_PVE];
        acc(kk).L   = [acc(kk).L;   repmat(Li, numel(row), 1)];
    end
end

%% ===== Compute per-layer proportions for each percentile =====
prop_by_layer = nan(numel(layers), K);   % rows: layers, cols: percentiles
for kk = 1:K
    for li = 1:numel(layers)
        mask = (acc(kk).L == li);
        if any(mask)
            prop_by_layer(li, kk) = mean(acc(kk).PVE(mask) > acc(kk).EE(mask));
        end
    end
end

if include_best_fit
    prop_best = nan(numel(layers),1);
    for li = 1:numel(layers)
        mask = (acc_best.L == li);
        if any(mask)
            prop_best(li) = mean(acc_best.PVE(mask) > acc_best.EE(mask));
        end
    end
end

%% ===== Pretty print as a table =====
pct_names = strcat("pct_", strrep(string(pct_list),'.','_'));  % e.g., pct_0_1
T = array2table(prop_by_layer, 'VariableNames', cellstr(pct_names), 'RowNames', cellstr(layers));
if include_best_fit
    T.best = prop_best;
end
disp('Proportion (Off->PV > On->PV) by layer:');
disp(T);

%% ===== Scatter plots (colored by layer) =====
if make_plots
    for kk = 1:K
        fig = figure('Color','w','Units','inches','Position',[0 0 6 5], 'Visible','on');  % set 'off' for silent batch
        hold on; grid on;

        % plot points by layer for consistent legend & color
        for li = 1:numel(layers)
            mask = (acc(kk).L == li);
            if any(mask)
                scatter(acc(kk).EE(mask), acc(kk).PVE(mask), 20, ...
                        'filled', 'MarkerFaceColor', cmap(li,:), 'MarkerEdgeColor', 'none');
            end
        end

        % x = y reference
        allX = acc(kk).EE; allY = acc(kk).PVE;
        if ~isempty(allX)
            lims = [min([allX; allY]) max([allX; allY])];
            if ~isfinite(lims(1)) || ~isfinite(lims(2)), lims = [0 1]; end
            plot(lims, lims, 'k--', 'LineWidth', 1);
            xlim(lims); ylim(lims);
        end
        xlabel('On \rightarrow PV (index 2)');
        ylabel('Off \rightarrow PV (index 4)');
        title(sprintf('Top %.3g%% by loss (all cells combined)', pct_list(kk)));

        % legend
        h = gobjects(numel(layers),1);
        for li = 1:numel(layers)
            h(li) = plot(nan, nan, 'o', 'MarkerFaceColor', cmap(li,:), 'MarkerEdgeColor', 'none', ...
                         'MarkerSize', 8, 'DisplayName', layers(li));
        end
        legend(h, 'Location', 'bestoutside', 'Box','off');

        % ----- save as PNG -----
        pct_tag = strrep(sprintf('%.3g', pct_list(kk)), '.', 'p');  % e.g., 0.1 -> "0p1"
        fname = sprintf('pv_gt_ee_top_%spct.png', pct_tag);
        exportgraphics(fig, fullfile(outdir, fname), 'Resolution', 300);
        % close(fig);  % uncomment to close after saving during batch runs
    end

    if include_best_fit && ~isempty(acc_best.EE)
        fig = figure('Color','w','Units','inches','Position',[0 0 6 5], 'Visible','on');  % set 'off' if you like
        hold on; grid on;

        for li = 1:numel(layers)
            mask = (acc_best.L == li);
            if any(mask)
                scatter(acc_best.EE(mask), acc_best.PVE(mask), 24, ...
                        'filled', 'MarkerFaceColor', cmap(li,:), 'MarkerEdgeColor', 'none');
            end
        end

        allX = acc_best.EE; allY = acc_best.PVE;
        lims = [min([allX; allY]) max([allX; allY])];
        if ~isfinite(lims(1)) || ~isfinite(lims(2)), lims = [0 1]; end
        plot(lims, lims, 'k--', 'LineWidth', 1);
        xlim(lims); ylim(lims);
        xlabel('On \rightarrow PV (index 2)');
        ylabel('Off \rightarrow PV (index 4)');
        title('Best fit per cell (min loss; ties included)');

        h = gobjects(numel(layers),1);
        for li = 1:numel(layers)
            h(li) = plot(nan, nan, 'o', 'MarkerFaceColor', cmap(li,:), 'MarkerEdgeColor', 'none', ...
                         'MarkerSize', 8, 'DisplayName', layers(li));
        end
        legend(h, 'Location', 'bestoutside', 'Box','off');

        % ----- save as PNG -----
        exportgraphics(fig, fullfile(outdir, 'pv_gt_ee_best_fit.png'), 'Resolution', 300);
        % close(fig);  % optional
    end
end
