function [perf,fr,spks] = postProcessData_new(data,s,trialStart,trialEnd,configName,options)
% calculate performance and FR for *a single spot on the spatial grid*
% input:
%   data structure with length = #sims, containing voltage information of
%   [time x spatial channel]. Time dimension contains information from each
%   spot in the spatial grid. Extract this information with
%   options.trialStart and options.trialEnd.
% output:
%   data
%       spatial channels,trials,time
% no plotting function for now

time_end = options.time_end;
plot_rasters = options.plotRasters;
fields = fieldnames(data);
ind = find(contains(fields,'_trial'),1);
jump = length(find([data.(fields{ind})]==1));  % number of variations/parameter sets
numTrials = length(data)/jump; %usually, 20 trials
% numChannels = 4;

% network properties
popNames = {s.populations.name};
popSizes = [data(1).model.specification.populations.size];
nPops = numel(popNames);
fieldNames = strcat(popNames,'_V_spikes');

% for this trial
tstart = trialStart;
tend = trialEnd;

[y1,fs] = audioread('200k_target1.wav');
y2 = audioread('200k_target2.wav');
t = (0:(length(y1)-1))/fs;

% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end
if plot_rasters, figure; end


for vv = 1:jump % for each varied parameter
    subData = data(vv:jump:length(data)); %grab data for this param variation
    try
        variedParamVal = mode([subData.(options.variedField)]); % should all be the same
    catch
        variedParamVal = 0;
    end
    
    for currentPop = 1:nPops
        
        if popSizes(currentPop) > 1 && ~isempty(options.chansToPlot)
            chansToPlot = options.chansToPlot;
        else
            chansToPlot = 1:popSizes(currentPop);
        end

        % skip processing for current population if not within specified subpopulation.
        if ~contains(popNames(currentPop),options.subPops), continue; end
        
        % for each spatial channel
        for channelNum = 1:popSizes(currentPop)
            channel = struct();
            
            % for each trial
            for trial = 1:numTrials
                channel(channelNum).popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(tstart:tend,channelNum);
            end
            spks.(popNames{currentPop})(vv).(['channel' num2str(channelNum)]) = channel(channelNum).popSpks;
            
            figName = sprintf('%s_%s%.03f_%s_channel%i_%s',configName,options.variedField,variedParamVal,popNames{currentPop},channelNum,num2str(vv));
            
            if strcmp(popNames{currentPop},'IC')  && vv > 1
                plot_rasters_final = 0;
            elseif ismember(channelNum,chansToPlot)
                plot_rasters_final = plot_rasters;
            else
                plot_rasters_final = 0;
            end
            
            [perf.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv),...
                fr.(popNames{currentPop}).(['channel' num2str(channelNum)])(vv)] = ...
                calcPCandPlot(channel(channelNum).popSpks,time_end,1,numTrials,plot_rasters_final,y1,y2,t,figName);
            
        end
    end
end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,plot_rasters,y1,y2,t,figName)

PCstr = '';

if calcPC
    % spks to spiketimes in a cell array of 20x2
    spkTime = cell(numTrials,1);
    for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
    spkTime = reshape(spkTime,numTrials/2,2);
    
    input = reshape(spkTime,1,numTrials);
    STS = SpikeTrainSet(input,250*10,(250+2986)*10);
    distMat = STS.SPIKEdistanceMatrix(250*10,(250+2986)*10);
    
    performance = calcpcStatic(distMat, numTrials/2, 2, 0);
    pc = mean(max(performance));
    PCstr = ['PC = ' num2str(pc)];
end

fr = 1000*mean(sum(raster(:,[2500:32500]),2))/3000;

%plot
x = 0.86;
y_raster = 0.4;
y_stim = 0.08;
y_psth = 0.12;
dy = 0.01;
x0 = 0.1;

if plot_rasters
    clf
    
    ypos = 0.91 - y_stim;
    subplot('position',[x0 ypos x y_stim]); % target 2
    plot(t,y2,'k'); xlim([0 time_end/1000]);
    title({PCstr,['FR = ' num2str(fr)]}); set(gca,'xtick',[],'ytick',[])
    
    ypos = ypos - dy - y_psth;
    subplot('position',[x0 ypos x y_psth]);
    plotPSTH(raster(numTrials/2+1:end,:)); set(gca,'xtick',[])
    
    ypos = ypos - dy - y_raster;
    subplot('position',[x0 ypos x y_raster]);
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    xlim([0 time_end*10]);
    line([0,time_end*10],[numTrials/2 + 0.5,numTrials/2 + 0.5],'color',[0.3 0.3 0.3]); set(gca,'xtick',[],'ytick',[])
    
    ypos = ypos - dy - y_psth;
    subplot('position',[x0 ypos x y_psth]);
    plotPSTH(raster(1:numTrials/2,:)); set(gca,'xtick',[])

    ypos = ypos - dy - y_stim;
    subplot('position',[x0 ypos x y_stim]); % target 1
    plot(t,y1,'k'); xlim([0 time_end/1000]); set(gca,'ytick',[]); 
    
    saveas(gcf,[figName '.png'])
end
    
end

function plotPSTH(raster)

t_vec = (0:200:35000)/10000;
[~,temp] = find(raster);
psth = histcounts(temp/10000,t_vec);
psth(end+1) = 0;

bar(t_vec,psth,'k');

lim = 30;
if max(psth,[],'all') > lim
   lim = lim*ceil(max(psth,[],'all')/lim); 
end

xlim([t_vec(1) t_vec(end)]); ylim([0 lim]);
ylabel('Spike count');
set(gca,'ytick',0:lim/3:lim);

end