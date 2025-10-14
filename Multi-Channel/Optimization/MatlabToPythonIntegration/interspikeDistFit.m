%clear all
%close all
%Calculate the interspike interval distribution

%The goal here is to examine and fit the interspike interval distribution of the data.


%Bring in data
addpath('results\')
filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/picture_fit56.mat";
data = load(filename).picture;

%PSTH + VR 100ms
%m = matfile('run_2025-09-11_20-28-42.mat');
%m = matfile('run_2025-09-24_12-23-46.mat');
%m = matfile('run_2025-09-24_14-40-11.mat');
%m = matfile('run_2025-09-24_16-14-45.mat');
%m = matfile('run_2025-09-24_17-08-02.mat');

%m = matfile('run_2025-09-29_13-19-42.mat');


%m = matfile('run_2025-09-29_13-41-22.mat');

%m = matfile('run_2025-09-29_14-23-42.mat');

%m = matfile('run_2025-09-29_15-25-13.mat');

%m = matfile('run_2025-09-29_21-58-21.mat');
%m = matfile('run_2025-10-01_00-08-37.mat');
%m = matfile('run_2025-10-01_07-08-28.mat');

%Layer 4
%m = matfile('run_2025-10-01_17-07-52.mat');


%%make_raster(m,data)

%m = matfile('run_2025-10-01_17-48-11.mat');

%make_raster(m,data)

%m = matfile('run_2025-10-01_18-53-40.mat');

%make_raster(m,data)

m = matfile('run_2025-10-01_23-08-57.mat');

outputs = m.output;
losses = m.losses;
params = m.param_tracker;

%Calculate the isi for all spikes accross all trials
epochnum = size(params);
%[min_val, min_idx] = min(losses(epochnum(1),:));
[min_val, min_idx] = min(losses(epochnum(1),2,:));
output_trial = min_idx;


isi_holder_data = isi_calc(data);
isi_holder_sim = isi_calc(outputs(:,:,output_trial));


figure;
plot(movmean(histcounts(isi_holder_data,1:10000)/sum(histcounts(isi_holder_data,1:10000)),100),'LineWidth',2); hold on
plot(movmean(histcounts(isi_holder_sim,1:10000)/sum(histcounts(isi_holder_sim,1:10000)),100),'LineWidth',2);
xlim([0,2000])
legend({'data','sim'})
xlabel('samples in samples/0.1ms')

figure;
hist(isi_holder_data,1000); hold on
hist(isi_holder_sim,1000);

%h = findobj(gca,'Type','patch');
%h.FaceColor = 05;
%h.EdgeColor = 'w';
%xlim([0,500])


disp('Mean Iterval Data < 200ms = ' + string(mean(isi_holder_data(find(isi_holder_data < 2000)))*0.1)+ ' ms')  %dt = 0.1ms
disp('Mean Iterval sim < 200ms = ' + string(mean(isi_holder_sim(find(isi_holder_sim < 2000)))*0.1)+ ' ms')  %dt = 0.1ms
% 
disp('Mean Iterval Data Total = ' + string(mean(isi_holder_data)*0.1)+ ' ms')  %dt = 0.1ms
disp('Mean Iterval sim Total = ' + string(mean(isi_holder_sim)*0.1)+ ' ms')  %dt = 0.1ms

% ---- Inputs ----
% isi_holder_data, isi_holder_sim  % in SAMPLES
dt_ms    = 0.1;          % your dt = 0.1 ms/sample
xmax_ms  = 200;          % display range (0–200 ms)
zoom_ms  = 15;           % 150 samples = 15 ms
bin_w_ms = 0.5;          % bin width (0.5 ms)

% ---- Convert to milliseconds ----
data_ms = isi_holder_data .* dt_ms;
sim_ms  = isi_holder_sim  .* dt_ms;

% ---- Common bins + histograms (PDF normalization) ----
edges = 0:bin_w_ms:xmax_ms;
centers = edges(1:end-1) + diff(edges)/2;

Nd = histcounts(data_ms, edges, 'Normalization','pdf');
Ns = histcounts(sim_ms,  edges, 'Normalization','pdf');

% ---- Summary stats ----
m_data_short = mean(data_ms(data_ms < zoom_ms), 'omitnan');
m_sim_short  = mean(sim_ms(sim_ms   < zoom_ms), 'omitnan');
m_data_full  = mean(data_ms, 'omitnan');
m_sim_full   = mean(sim_ms,  'omitnan');

% Optional two-sample KS test (remove NaNs)
try
    [~,p_ks] = kstest2(data_ms(~isnan(data_ms)), sim_ms(~isnan(sim_ms)));
catch
    p_ks = NaN;
end

% ---- Figure styling ----
f = figure('Color','w','Units','pixels','Position',[100 100 850 750]);
tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');

% ===== Panel 1: Full range (log y) =====
ax1 = nexttile;
hold(ax1,'on');
% Light shading for the <15 ms region
yl = [1e-5 1]; % provisional, updated after plotting
patch(ax1,[0 zoom_ms zoom_ms 0],[1e-3 1e-3 1 1],[0.92 0.92 0.92], ...
      'EdgeColor','none','FaceAlpha',0.35);
% Stairs plots (crisp overlay)
stairs(ax1, centers, Nd, 'LineWidth',2.0, 'Color',[0.20 0.20 0.80]);
stairs(ax1, centers, Ns, 'LineWidth',2.0, 'Color',[0.10 0.10 0.10]);

set(ax1,'YScale','log');
xlim(ax1,[0 xmax_ms]);
ylim(ax1,'auto'); yl = ylim(ax1); % update after data drawn
% Re-draw shaded patch to full y-lims
delete(findobj(ax1,'Type','patch'));
patch(ax1,[0 zoom_ms zoom_ms 0],[yl(1) yl(1) yl(2) yl(2)], ...
      [0.92 0.92 0.92],'EdgeColor','none','FaceAlpha',0.35);

xline(ax1, zoom_ms,'k--','150 samples (15 ms)', ...
      'LabelHorizontalAlignment','left','LabelVerticalAlignment','middle','LineWidth',1);

grid(ax1,'on');
xlabel(ax1,'Inter-spike interval (ms)');
ylabel(ax1,'Probability density (log scale)');
title(ax1,'ISI distributions: Data vs. Simulation');
legend(ax1,{'Data','Simulation'},'Location','northeast','Box','off');

% ===== Panel 2: Zoom into <15 ms =====
ax2 = nexttile; hold(ax2,'on');
bar(ax2, centers, Nd, 1.0,'FaceAlpha',0.35,'EdgeColor','none','FaceColor',[0.25 0.25 0.85]);
bar(ax2, centers, Ns, 1.0,'FaceAlpha',0.35,'EdgeColor','none','FaceColor',[0.10 0.10 0.10]);
xlim(ax2,[0 zoom_ms]);
grid(ax2,'on');
xlabel(ax2,'Inter-spike interval (ms)');
ylabel(ax2,'Probability density');
legend(ax2,{'Data','Simulation'},'Location','northeast','Box','off');

% ---- Global title + annotation ----
sgtitle(sprintf('Tail largely matches; simulation compressed for ISIs < %.0f samples (%.0f ms)', ...
        zoom_ms/dt_ms, zoom_ms));

stat_str = sprintf(['Means:  <%.0f ms  Data = %.2f ms, Sim = %.2f ms\n',...
                    '        Full     Data = %.2f ms, Sim = %.2f ms',...
                    '%s'], ...
                    zoom_ms, m_data_short, m_sim_short, m_data_full, m_sim_full, ...
                    iff(isnan(p_ks),'',sprintf('\nKS test p = %.3g',p_ks)));
%annotation(f,'textbox',[0.63 0.01 0.35 0.12],'String',stat_str, ...
%           'EdgeColor','none','FontName','Helvetica','FontSize',10,'HorizontalAlignment','right');

% ---- Consistent typography ----
set(findall(f,'-property','FontName'),'FontName','Helvetica');
set(findall(f,'-property','FontSize'),'FontSize',12);



%Isi gathering function
function isi_holder = isi_calc(data)
    isi_holder = [];

    for k = 1:10
        isi_holder = [isi_holder, diff(find(data(k,:)))];
    end
end


function y = iff(cond,a,b), if cond, y=a; else, y=b; end, end