function plotPerfvsParams(pop,data,varies,simDataDir,configName)

% 12-01-23: only works on single-config, need to change

% numVaries = length(snn_out)/20;
% 
% fields = fieldnames(snn_out);
% temp = snn_out(1).varied; temp([1 2]) = [];

variedParams = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));
variedParams(variedParams == 1) = [];
params = {varies(variedParams).range};
paramNames = {varies(variedParams).param};
paramConxns = {varies(variedParams).conxn};

x = params{1};
y = params{2};

nCells = numel(fieldnames(data.perf.(pop)));

for ch = 1:nCells
    
    perf = reshape(data.perf.(pop).(['channel' num2str(ch)]),[numel(y) numel(x)]);
    
    figure('unit','inches','position',[4 4 11 4.5]);
    ax1 = subplot(1,2,1);
    imagesc(x,y,perf);
    xlabel([paramConxns{1}  '_{' paramNames{1} '}']);
    ylabel([paramConxns{2}  '_{' paramNames{2} '}']);
    set(gca,'ydir','normal','xtick',x,'ytick',y,'fontsize',10);
    title(['Performance vs. ' paramNames{1}]);
    cc = colorbar; cc.Label.String = 'Performance';
    clim([50 max(perf,[],'all')]);
    
    FR = reshape(data.fr.(pop).channel1,[numel(y) numel(x)]);
    
    ax2 = subplot(1,2,2);
    imagesc(x,y,FR);
    xlabel([paramConxns{1} '_{' paramNames{1} '}']);
    ylabel([paramConxns{2} '_{' paramNames{2} '}']);
    set(gca,'ydir','normal','xtick',x,'ytick',y,'fontsize',10);
    title(['FR vs. ' paramNames{1}]);
    cc = colorbar; cc.Label.String = 'Firing rate (Hz)';
    colormap(ax2,'gray');
    clim([0 max(FR,[],'all')]);
    titleStr = [pop ', CH' num2str(ch) ', ' configName];
    sgtitle(titleStr);
    
    saveas(gcf,[simDataDir filesep titleStr ' grid search.png']);
    savefig(gcf,[simDataDir filesep titleStr ' grid search']);
    
end

end