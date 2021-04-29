function plotPerformanceGrids_v3(data,s,subPops,targetIdx,mixedIdx,simOptions)
% plot performance grids for specified subpopulation of neurons
%
% v3 changes the script into a function

varied_param = simOptions.varied_param;
varies = simOptions.varies;
expVar = simOptions.expVar;
subz = simOptions.subz;
locationLabels = simOptions.locationLabels;

% setup populations
popNames = {s.populations.name};
% subPops = popNames(~contains(popNames,'TD') & ~contains(popNames,'X')); %remove TD neuron
popNamesT = strcat({s.populations.name},'T');
popNamesM = strcat({s.populations.name},'M');
numPops = numel(subPops);
popSizes = [s.populations.size];
% popSizes = [snn_out(1).model.specification.populations.size];
popSizes = popSizes(contains(popNames,subPops));
onlyC = contains(subPops,'C') & length(subPops) == 1;

% chanLabel = {'left sig','gauss','U','right sig'};
chanLabel = simOptions.chanLabels;
for vv = 1:length(varies(varied_param).range)
    % simulation annotation
    annotTable = [{varies.conxn}' {varies.param}' {varies.range}'];
    
    % if no parameter is varied
    if varied_param == 1
        % remove "R-C gSYN", trials
        RCgSYN_idx = find(contains([{varies.conxn}'],'R->C') & contains([{varies.param}'],'gSYN'));
        annotTable([RCgSYN_idx; varied_param],:, :) = []; 
        
        annotStr = strcat(annotTable(:,1), {' '}, annotTable(:,2), {' = '}, strtrim(num2str([annotTable{:,3}]')));
        annotStr{end+1} = sprintf('RC-gSYN = [%.02f %.02f %.02f %.02f]', [varies(RCgSYN_idx).range]);
    else
        % remove "R-C gSYN", varied parameter, trials
        RCgSYN_idx = find(contains([{varies.conxn}'],'R->C') & contains([{varies.param}'],'gSYN'));
        annotTable([RCgSYN_idx;; varied_param],:, :) = [];
        annotTable(1,:,:) = [];
        
        currentVariedValue = varies(varied_param).range(vv);
        annotStr = strcat(annotTable(:,1), {' '}, annotTable(:,2), {' = '}, strtrim(num2str([annotTable{:,3}]')));
        annotStr{end+1} = [expVar ' = ' num2str(currentVariedValue)];
        annotStr{end+1} = sprintf('RC-gSYN = [%.02f %.02f %.02f %.02f]', [varies(RCgSYN_idx).range]);
    end
      
    % figure setup
    figwidth = 900; %300 * number of neurons?
    figheight = min(length(popSizes)*300,800);
    h = figure('position',[200 50 figwidth figheight]);
    plotwidth = 0.233; % percentage, 0.7*num neurons
    plotheight = 1/numPops*0.55; % percentage
    xstart = 0.1;
    ystart = 0.1;
    for pop = 1:numPops
        for chan = 1:popSizes(pop)

            % subfigure positions
            col = chan-1;
            row = pop-1;
            xoffset = xstart+plotwidth*(col)*1.2;
            yoffset = ystart+plotheight*1.6*row;

    %         subplot('Position',[xoffset yoffset plotwidth plotheight])
    %         subplot('Position',[xoffset yoffset+plotheight+0.01 plotwidth plotheight*0.4])

            % mixed cases
            if sum(ismember(subz,mixedIdx)) > 0
                for i = 1:length(mixedIdx)
                    idx = (mixedIdx(i) == subz);
                    perf.(subPops{pop})(i) = data(idx).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                    fr.(subPops{pop})(i) = data(idx).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                end
                subplot('Position',[xoffset yoffset plotwidth plotheight])
                plotPerfGrid(perf.(subPops{pop})',fr.(subPops{pop})',[]);
                
                % add axes
                if sum(onlyC) || (chan==1 && pop==1)
                    xticks(1:4); xticklabels(locationLabels)
                    yticks(1:4); yticklabels(fliplr(locationLabels))
                    xlabel('target location')
                    ylabel('masker location')
                end
            end

            % target or masker only cases
            if sum(ismember(subz,targetIdx)) > 0
                perf.(popNamesT{pop}) = zeros(1,4);
                perf.(popNamesM{pop}) = zeros(1,4);
                fr.(popNamesT{pop}) = zeros(1,4);
                fr.(popNamesM{pop}) = zeros(1,4);
                if ~isempty(targetIdx)
                    for i = 1:length(targetIdx)
                        idx = (targetIdx(i) == subz);
                        perf.(popNamesT{pop})(i) = data(idx).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                        fr.(popNamesT{pop})(i) = data(idx).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                    end
                end
                if exist('maskerIdx','var')
                    if ~isempty(maskerIdx)
                        for i = 1:length(maskerIdx)
                            idx = (maskerIdx(i) == subz);
                            perf.(popNamesM{pop})(i) = data(i).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                            fr.(popNamesM{pop})(i) = data(i).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                        end
                    end
                end
                subplot('Position',[xoffset yoffset+plotheight+0.01 plotwidth plotheight*0.4])
                if pop == 1
                    plotPerfGrid([perf.(popNamesT{pop});perf.(popNamesM{pop})],[fr.(popNamesT{pop});fr.(popNamesM{pop})],chanLabel{chan});
                else
                    plotPerfGrid([perf.(popNamesT{pop});perf.(popNamesM{pop})],[fr.(popNamesT{pop});fr.(popNamesM{pop})],'');
                end

                if chan == 1, ylabel(subPops{pop}); end
            end
        end
    end
    % simulation info
    annotation('textbox',[xoffset+plotwidth*1.2 yoffset+plotheight*0.5 plotwidth plotheight],...
           'string',annotStr,...
           'FitBoxToText','on',...
           'LineStyle','none')
end