function plotPerfGrid(neuronPerf,neuronFR,titleString,textColorThresh)
% plotPerfGrid(neuronPerf,neuronFR,titleString,textColorThresh)
%
% plots performance grids; can handle a data array of 16 or 8 elements
% works with mouseNetwork_main
if ~exist('textColorThresh','var'), textColorThresh = 70; end
if ~exist('neuronFR','var'), neuronFR = 'n/a'; end

% 2D plot (imagesc)
if numel(neuronPerf) == 16
    % 4x4 grid
    [X,Y] = meshgrid(1:4,4:-1:1);
    str = cellstr(num2str(round(neuronPerf)));
    str2 = cellstr(num2str(round(neuronFR(:))));
    neuronPerf = reshape(neuronPerf,4,4);
    imagesc(flipud(neuronPerf));
    xticks([1:4]); xticklabels([])
    yticks([1:4]); yticklabels([])
elseif numel(neuronPerf) == 8
    % 2x4 grid
    [X,Y] = meshgrid(1:4,1:2);
    str = cellstr(num2str(round(neuronPerf(:))));
    str2 = cellstr(num2str(round(neuronFR(:))));
    imagesc(neuronPerf);
    xticks([]);
    yticks(1:2); yticklabels({'target only','masker only'})
    ytickangle(60)
end

title(titleString)
colormap('parula');
t = text(X(:),Y(:)-0.15,str,'Fontsize',12,'HorizontalAlignment', 'Center');
t2 = text(X(:),Y(:)+0.15,strcat('(',str2,')'),'Fontsize',12,'HorizontalAlignment', 'Center');
for i= 1:numel(neuronPerf)
    if neuronPerf(i)>textColorThresh
        t(i).Color = 'k';
        t2(i).Color = 'k';
    else
        t(i).Color= 'w';
        t2(i).Color= 'w';
    end
end
caxis([50,100])
set(gca,'fontsize',12)