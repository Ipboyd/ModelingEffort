% network output to performance grid
% - spike generator
% - reshape
% - performance grid
%
% load output:

load('training_set_1.mat','output_training')

%% FR to spikes
addpath('C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\ICSimStim')
addpath('C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\plotting')
fr = output_training;
t_vec = 1:1:length(fr);
[spike_train,~] = spike_generator_rr(fr,t_vec);
spike_train = spike_train';

%% reshape
spike_train_reshaped = reshape(spike_train,[],20);
plotSpikeRasterFs(logical(spike_train_reshaped'),'PlotType','vertline2','Fs',1);
xlim([0 48000])

%
frr = reshape(fr,[],20);
imagesc(frr')

%% Calcualte Performance
time_end = 2000;
numTrials = 20;
plot_rasters = 0;
for config = 1:24
    tstart = 1+2000*(config-1);
    tend = tstart+2000-1;
    currentConfigSpks = Cspks(tstart:tend,:);
    [pre.performance(config), pre.frate(config)]=calcPCandPlot(currentConfigSpks',time_end,1,numTrials,plot_rasters,'hi');
    
    currentConfigSpks = spike_train_reshaped(tstart:tend,:);
    [post.performance(config), post.frate(config)]=calcPCandPlot(currentConfigSpks',time_end,1,numTrials,plot_rasters,'hi');
end

% plot performance grid
maskerIdx = 1:4;
targetIdx = 5:5:24;
mixedIdx = 1:24;
mixedIdx([maskerIdx targetIdx]) = [];
figure;
plotPerfGrid(pre.performance(mixedIdx),pre.frate(mixedIdx),'perf grid; from smoothed output');
figure;
plotPerfGrid(post.performance(mixedIdx),post.frate(mixedIdx),'perf grid; from smoothed output');

%% compare PSTH
for i = 1:20
    spkTimesPre{i} = find(Cspks(:,i));
    spkTimesPost{i} = find(spike_train_reshaped(:,i));
end
figure;
h = subplot(4,1,1)
temp = spkTimesPre(1:10);
psth(vertcat(temp{:}),100,1000,10,2000*24,h)
title('song 1, dynasim output')
h = subplot(4,1,2)
temp = spkTimesPre(11:20);
psth(vertcat(temp{:}),100,1000,10,2000*24,h)
title('song 2, dynasim output')
h = subplot(4,1,3)
temp = spkTimesPost(1:10);
psth(vertcat(temp{:}),100,1000,10,2000*24,h)
title('song 1, recovered output')
h = subplot(4,1,4)
temp = spkTimesPost(11:20);
psth(vertcat(temp{:}),100,1000,10,2000*24,h)
title('song 2, recovered output')

%% auxiliary function
function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,plot_rasters,figName)

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
    clf
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    title({PCstr,['FR = ' num2str(fr)]});
    xlim([0 time_end])
    line([0,time_end],[numTrials/2 + 0.5,numTrials/2 + 0.5],'color',[0.3 0.3 0.3])
    saveas(gcf,[figName '.tif'])
end
    
end