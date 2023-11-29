function [perf,fr,spks] = plotPSTHTree(snn_out,s,tstart,tend,configName,options)
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
SpatAttention = options.SpatialAttention;

% network properties
popNames = {s.populations.name};
nPops = numel(popNames); 
fieldNames = strcat(popNames,'_V_spikes');

nChans = size(snn_out(1).R1On_V,2);

if isempty(options.locNum), annotConfig = configName(end-7:end-4);
else, annotConfig = configName(end-3:end);
end

% visualize spikes for specified populations
if ~isfield(options,'subPops'), options.subPops = popNames; end

% in same order as popNames
if nPops == 8% no top down
    subplot_locs = [11 12 7 9 8 4 5 2]; %[11 7 8 4 5 9 6 2];
elseif nPops == 9% no top down
    subplot_locs = [11 10 7 8 4 5 9 6 2];
elseif nPops == 10 % no C neuron
    subplot_locs = [14 16 10 12 9 11 6 8 5 7 ];
else
    subplot_locs = [14 10 11 7 8 12 9 1 2 5];%[14 13 10 11 7 8 12 9 1 2 5];
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
        
        figure('unit','inches','position',[6 3 6 5]);
        
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
            % spks.(popNames{currentPop})(vv).channel = popSpks;
            
            [perf.(popNames{currentPop}).channel(vv),...
                fr.(popNames{currentPop}).channel(vv)] = ...
                calcPCandPlot(popSpks,time_end,1,numTrials,popNames{currentPop},...
                subplot_locs(currentPop),SpatAttention);
            
        end
        
        figName = sprintf('%s_CH%f_set%s_PSTH',configName,ch,num2str(vv));
        
        annotation('textbox',[.1 .82 .1 .1], ...
            'String',[annotConfig, ', CH' num2str(ch) ],'EdgeColor','none','FontSize',20)
        
        saveas(gcf,[figName '.png']);
        savefig(gcf,[figName '.fig']);
%         clf;
    end

end

end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,numTrials,unit,subplot_loc,SpatAttention)

PCstr = '';

if calcPC
    % spks to spiketimes in a cell array of 20x2
    spkTime = cell(numTrials,1);
    for ii = 1:numTrials, spkTime{ii} = find(raster(ii,:)); end
    spkTime = reshape(spkTime,numTrials/2,2);
    
    input = reshape(spkTime,1,numTrials);
    STS = SpikeTrainSet(input,300*10,(300+3000)*10);
    distMat = STS.SPIKEdistanceMatrix(300*10,(300+3000)*10);
    
    performance = calcpcStatic(distMat, numTrials/2, 2, 0);
    pc = mean(max(performance));
    PCstr = ['PC = ' num2str(round(pc))];
end

% fr = round(1000*mean(sum(raster(:,3000:33000),2))/3000);
fr = round(1000*mean(sum(raster(:,3000:18000),2))/1500);

% ind2sub counts down per column first, 
if SpatAttention
    [c,r] = ind2sub([3 5],subplot_loc);
    r = 6-r;
        ypos = 0.06 + 0.2*(r-1);
y = 0.10;

else
    [c,r] = ind2sub([4 4],subplot_loc);
    r = 5-r;
    ypos = 0.06 + 0.25*(r-1);
y = 0.12;

end

xpos = 0.06 + 0.23*(c-1);
x = 0.22;

subplot('position',[xpos ypos x y]); hold on;

% plot %target 1
t_bin = 200; % 20 ms = 200 indices
t_vec = 0:t_bin:(time_end*10);
colors = {'k','r'};
for tid = 1:2
    spkTimeIDs = cellfun(@(x) histcounts(x,t_vec),spkTime(:,tid),'UniformOutput',false);
    PSTH = sum(vertcat(spkTimeIDs{:}));
    plot(t_vec(1:end-1),PSTH,colors{tid});
end
xlim([0 time_end*10]);
title({unit,PCstr,['FR = ' num2str(fr)]}); set(gca,'xtick',[],'ytick',[])

end
