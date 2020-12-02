% plot performance grids for specified subpopulation of neurons

% cleanup
clear perf fr

% performance vector has dimensions [numSpatialChan,nvaried]
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

% indices for target-masker configuraitons
fileNames = {ICfiles.name};
targetIdx = find(contains(fileNames,'m0') & contains({ICfiles.name},'_E')); %target only
maskerIdx = find(contains(fileNames,'s0') & contains({ICfiles.name},'_E')); %masker only
mixedIdx = find(~contains(fileNames,'m0') & ~contains(fileNames,'s0') & contains({ICfiles.name},'_E'));

% setup populations
popNames = {s.populations.name};
% subPops = popNames(~contains(popNames,'TD') & ~contains(popNames,'X')); %remove TD neuron
popNamesT = strcat({s.populations.name},'T');
popNamesM = strcat({s.populations.name},'M');
subPops = options.subPops;
numPops = numel(subPops);
popSizes = [temp(1).model.specification.populations.size];
popSizes = popSizes(contains(popNames,subPops));

for vv = 1:length(varies(varied_param).range)
    % simulation annotation
    currentVariedValue = varies(varied_param).range(vv);
    annotStr = [{varies.conxn}' {varies.param}' {varies.range}'];
    annotStr([1:2 3:10 varied_param],:) = [];
    annotStr = strcat(annotStr(:,1), {' '}, annotStr(:,2), {' = '}, strtrim(num2str([annotStr{:,3}]')));
    annotStr{end+1} = sprintf('RC-gSYN = [%.02f %.02f %.02f %.02f]', [varies(3:6).range]);
    annotStr{end+1} = sprintf('InhR-gSYN = [%.02f %.02f %.02f %.02f]', [varies(7:10).range]);
    annotStr{end+1} = [expVar ' = ' num2str(currentVariedValue)];

    % figure setup
    figwidth = 1200;
    figheight = min(length(popSizes)*300,800);
    h = figure('position',[200 50 figwidth figheight]);
    plotwidth = 0.175; % percentage
    plotheight = 1/numPops*0.6; % percentage
    xstart = 0.1;
    ystart = 0.05;
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
                if ~isempty(maskerIdx)
                    for i = 1:length(maskerIdx)
                        idx = (maskerIdx(i) == subz);
                        perf.(popNamesM{pop})(i) = data(i).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                        fr.(popNamesM{pop})(i) = data(i).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                    end
                end    
                subplot('Position',[xoffset yoffset+plotheight+0.01 plotwidth plotheight*0.4])
                plotPerfGrid([perf.(popNamesT{pop});perf.(popNamesM{pop})],[fr.(popNamesT{pop});fr.(popNamesM{pop})],subPops{pop});

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