%Layer 4
losses_all = [];
addpath('results\')
m = matfile('run_2025-10-03_10-31-18.mat');
m2 = matfile('run_2025-10-03_10-47-06.mat'); 
m3 = matfile('run_2025-10-03_11-10-17.mat');

losses_all = [losses_all;m.losses(:,2,1)];
losses_all = [losses_all;m2.losses(:,2,1)];
losses_all = [losses_all;m3.losses(:,2,1)];

losses_all = losses_all'

losses_all = losses_all(:);

grp  = categorical(repelem(["1 Layer","2 Layers","3 Layers"],30))';

% Assume you already built losses_all (Nx1) and grp (Nx1 categorical: A,B,C)
figure; boxchart(grp, losses_all);
hold on; grid on; set(gcf,'Color','w');
ylabel('Loss');
yticklabels('')

% Split by group
cats = categories(grp);
A = losses_all(grp == cats{1});
B = losses_all(grp == cats{2});
C = losses_all(grp == cats{3});

% Two-sample t-tests (unequal variances is also fine: 'Vartype','unequal')
[~,pAB] = ttest2(A,B);
[~,pAC] = ttest2(A,C);
[~,pBC] = ttest2(B,C);

% Simple multiple-comparison correction (Bonferroni)
pvec = [pAB pAC pBC];
pvec = min(1, 3*pvec);  % 3 tests → Bonferroni
[pAB,pAC,pBC] = deal(pvec(1),pvec(2),pvec(3));

% Add significance bars
yl = ylim; y0 = yl(2); step = range(yl)*0.06;
pairs = [1 2; 1 3; 2 3];
pvals = [pAB pAC pBC];

for k = 1:size(pairs,1)
    x = pairs(k,:);
    y = y0 + step*k;
    plot(x, [y y], 'k','LineWidth',1.2);                % horizontal bar
    plot([x(1) x(1)], [y-step*0.3 y], 'k','LineWidth',1.2);
    plot([x(2) x(2)], [y-step*0.3 y], 'k','LineWidth',1.2);
    text(mean(x), y + step*+0.55, p2stars(pvals(k)), ...
        'HorizontalAlignment','center','FontSize',12);
end
ylim([yl(1) y0 + step*(size(pairs,1)+1)]);

% --- helper (can live at end of your script) ---
function s = p2stars(p)
    if p < 1e-4, s = '****';
    elseif p < 1e-3, s = '***';
    elseif p < 0.01, s = '**';
    elseif p < 0.05, s = '*';
    else, s = 'n.s.';
    end
end