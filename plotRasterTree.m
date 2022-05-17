function [perf,fr,spks] = plotRasterTree(snn_out,s,tstart,tend,configName,options,model)
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
fields = fieldnames(snn_out);
ind = find(contains(fields,'_trial'),1);
jump = length(find([snn_out.(fields{ind})]==1));
numTrials = length(snn_out)/jump; % # total trials (20 for single parameter set)

% network properties
popNames = {s.populations.name};
nPops = numel(popNames); 
fieldNames = strcat(popNames,'_V_spikes');

nChans = size(snn_out(1).IC_V,2);


% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end

% in same order as popNames
if nPops == 15 % on and off columns
    subplot_locs = [20 13 14 7 8 15 9 23 16 17 10 11 18 12 3.5];
% elseif nPops > 6 % on and off columns, zero SOM neurons
%     subplot_locs = [14,9,10,5,6,16,11,12,7,8];
else % one column only with SOM neurons
    subplot_locs = [20 13 14 7 8 15 9 3.5];
    % subplot_locs = [14,9,10,5,6];
end
% locs = {'90','45','0','-90'};

for vv = 1:jump % for each varied parameter
    
    subData = snn_out(vv:jump:length(snn_out)); %grab data for this param variation
    
    try
        variedParamVal = mode([subData.(options.variedField)]); % should all be the same
    catch
        variedParamVal = 0;
    end
    
    for ch = 1:nChans
        
        figure('unit','inches','position',[6 3 9.5 5]);

        
        for currentPop = 1:nPops
            
            popSpks = [];
            
            % for each trial
            for trial = 1:numTrials
                
                if strcmp(fieldNames{currentPop},'C_V_spikes')
                    popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(tstart:tend);
                else
                    popSpks(trial,:) = subData(trial).(fieldNames{currentPop})(tstart:tend,ch);
                end
                
            end
            spks.(popNames{currentPop})(vv).channel = popSpks;
            
            [perf.(popNames{currentPop}).channel(vv),...
                fr.(popNames{currentPop}).channel(vv)] = ...
                calcPCandPlot(popSpks,time_end,1,numTrials,popNames{currentPop},...
                subplot_locs(currentPop));
            
        end
        
        figName = sprintf('%s_CH%f_set%s',configName,ch,num2str(vv));
        
        annotation('textbox',[.1 .82 .1 .1], ...
            'String',[configName(end-3:end), ', CH' num2str(ch) ],'EdgeColor','none','FontSize',20)
        
        saveas(gcf,[figName '.png']);
        savefig(gcf,[figName '.fig']);
%         clf;
    end

end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,unit,subplot_loc)

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
    PCstr = ['PC = ' num2str(round(pc))];
end

fr = round(1000*mean(sum(raster(:,2500:32500),2))/3000);

% ind2sub counts down per column first, 
[c,r] = ind2sub([6 4],subplot_loc);
r = 5-r;


xpos = 0.06 + 0.16*(c-1);
ypos = 0.06 + 0.25*(r-1);
x = 0.12;
y = 0.12;

subplot('position',[xpos ypos x y]);

% raster = raster(1:10,:);

% % plot only target 1 response
% plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
% xlim([0 time_end*10]); ylim([0.5 10.5]);
% title({unit,PCstr,['FR = ' num2str(fr)]}); set(gca,'xtick',[],'ytick',[])

% plot both targets
plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
xlim([0 time_end*10]); ylim([0.5 20.5]);
title({unit,PCstr,['FR = ' num2str(fr)]}); set(gca,'xtick',[],'ytick',[])
line([0 time_end*10],[10.5 10.5],'color','k');

end
