function loadPrevDynasimInputs
% load and analyze inputs used for previous simulations
root = 'Z:\eng_research_hrc_binauralhearinglab\kfchou\ActiveProjects\MiceSpatialGrids\ICStim\Bird\Sigma7VarySharpeningStrength\20190810-163645';
files = dir(root);
dirFlags = [files.isdir];
folders = {files(dirFlags).name}';
for i = 1:length(folders)-2
    load([root filesep folders{i+2} filesep 'solve' filesep 'IC_spks.mat'],'spks');

    time_end = size(spks,3);
    plot_rasters = 0;
    numTrials = 20;
    for j = 1:4
        Cspks = squeeze(spks(:,j,:));
        [data(i).perf.C(j),data(i).fr.C(j)] = calcPCandPlot(Cspks,time_end,1,plot_rasters,numTrials);     
    end
end

ICfiles = files(dirFlags(3:end));
subz = 1:24;
plotPerformanceGrids;
end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,plot_rasters,numTrials)

PCstr = '';

if calcPC
    % spks to spiketimes in a cell array of 20x2
    tau = linspace(1,30,100);
    spkTime = cell(numTrials,1);
    for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
    spkTime = reshape(spkTime,numTrials/2,2);
    % calculate distance matrix & performance
    distMat = calcvr(spkTime, tau);
    [performance, ~] = calcpc(distMat, numTrials/2, 2, 1,[], 'new');
    pc = mean(max(performance));
    PCstr = ['PC = ' num2str(pc)];
end

if size(raster,2) > 3000
fr = 1000*mean(sum(raster(:,[1:1000,4000:end]),2))/2000;
else
    fr = 1000*mean(sum(raster,2))/3000;
end
%plot
if plot_rasters
    figure;
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    title({PCstr,['FR = ' num2str(fr)]});
    xlim([0 time_end])
    line([0,time_end],[numTrials/2 + 0.5,numTrials/2 + 0.5],'color',[0.3 0.3 0.3])
end
    
end