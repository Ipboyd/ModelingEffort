function plotPerfvsParams(pop,data,varies,simDataDir)

% numVaries = length(snn_out)/20;
% 
% fields = fieldnames(snn_out);
% temp = snn_out(1).varied; temp([1 2]) = [];

variedParams = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));
variedParams(variedParams == 1) = [];
params = {varies(variedParams).range};
paramNames = {varies(variedParams).param};

x = params{1};
y = params{2};

perf = reshape(data.perf.(pop).channel1,[numel(y) numel(x)]);

figure('unit','inches','position',[4 4 11 4.5]);
subplot(1,2,1);
imagesc(x,y,perf);
xlabel(['Within-column_{' paramNames{1} '}']); ylabel(['Cross-column_{' paramNames{2} '}']);
set(gca,'ydir','normal','xtick',x,'ytick',y,'fontsize',10);
title(['Performance vs. ' paramNames{1}]);
cc = colorbar; cc.Label.String = 'Performance';

FR = reshape(data.fr.(pop).channel1,[numel(y) numel(x)]);

subplot(1,2,2);
imagesc(x,y,FR);
xlabel(['Within-column_{' paramNames{1} '}']); ylabel(['Cross-column_{' paramNames{2} '}']);
set(gca,'ydir','normal','xtick',x,'ytick',y,'fontsize',10);
title(['FR vs. ' paramNames{1}]);
cc = colorbar; cc.Label.String = 'Firing rate (Hz)';

sgtitle(pop);

saveas(gcf,[simDataDir filesep pop '_perf_FR_grid.png']);
savefig(gcf,[simDataDir filesep pop '_perf_FR_grid']);

end