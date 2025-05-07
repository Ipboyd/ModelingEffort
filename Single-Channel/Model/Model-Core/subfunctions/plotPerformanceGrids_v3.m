function [approximate_grid,fr_grid] = plotPerformanceGrids_v3(data, s, annotTable, subPops, targetIdx, mixedIdx, simOptions, expName, plot_all)
    % plot performance grids for specified subpopulation of neurons
    %
    % v3 changes the script into a function
    


    subz = simOptions.subz;
    locationLabels = simOptions.locationLabels;

    % setup populations
    popNamesT = {s.populations.name};
    numPops = numel(subPops);
    popSizes = [s.populations.size];
    popSizes = popSizes(matches(popNamesT, subPops));
    onlyC = any(contains(subPops, 'C')) & length(subPops) == 1;

    % rearrange subPops if C is not the only thing in subPops
    if ~onlyC
        C_ind = find(contains(subPops, 'C'));
        subPops = subPops([setdiff(1:length(subPops), C_ind), C_ind]);

        %Remove Ron from subpops so that we only plot C in order to save
        %time with GA 7/19 IB
       % subPops = {subPops{2}}
    end

    chanLabels = simOptions.chanLabels;

    % check if varied parameter is vector or matrix
    numVars = length(annotTable);
    
    approximate_grid = [];
    fr_grid = [];

    for vv = 1:numVars
        annotStr = annotTable{vv};

        % figure setup
        figwidth = max(length(popSizes) * 300, 900); % 300 * number of neurons?
        figheight = min(length(popSizes) * 300, 800);
        h = figure('position', [200 50 figwidth figheight]);
        %IB addeed for speed purposes
        if plot_all == 0
            set(h,'Visible','off');
        else
            
        end
        %set(h,'Visible','off');

        plotheight = 1 / numPops * 0.55; % percentage
        xstart = 0.1;
        ystart = 0.1;

    
        for pop = 1:numPops
            % vary grid width depending on the number of neurons in subpopulation
            if popSizes(pop) == 1
                plotwidth = 0.233; % percentage, 0.7*num neurons
            else
                plotwidth = 0.14;
            end

            if plot_all == 0
                pop_tracker = 1;
            else
                pop_tracker = popSizes(pop);
            end

            %for chan = 1:popSizes(pop)
            
            for chan = 1:pop_tracker %Changed 7/19 to just look for upper left corner IB
                
                % subfigure positions
                col = chan - 1;
                row = pop - 1;
                xoffset = xstart + plotwidth * (col) * 1.2;
                yoffset = ystart + plotheight * 1.6 * row;

                % mixed cases
                if sum(ismember(subz, mixedIdx)) > 0
                    for i = 1:length(mixedIdx)
                        idx = subz(mixedIdx(i) == subz);
                        perf.(subPops{pop})(i) = data(idx).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                        fr.(subPops{pop})(i) = data(idx).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                    end
                    ax = subplot('Position', [xoffset yoffset plotwidth plotheight]);
                    plotPerfGrid(perf.(subPops{pop})', fr.(subPops{pop})', []);
                    
                    

                    if popSizes(pop) == 1
                    %This will need to be changed if Ron this put back in.
                    approximate_grid = [approximate_grid;flipud(reshape(perf.(subPops{pop}),4,4))];
                    fr_grid = [fr_grid;flipud(reshape(fr.(subPops{pop}),4,4))];
                    end

                    % add axes
                    if onlyC || (chan == 1 && pop == 1)
                        xticks(1:4);
                        xticklabels(locationLabels);
                        yticks(1:4);
                        yticklabels(fliplr(locationLabels));
                        xlabel('target location');
                        ylabel('masker location');
                    end
                end

                % target-only configs
                if sum(ismember(subz, targetIdx)) > 0
                    perf.(popNamesT{pop}) = zeros(1, 4);
                    fr.(popNamesT{pop}) = zeros(1, 4);

                    if ~isempty(targetIdx)
                        for i = 1:length(targetIdx)
                            idx = subz(targetIdx(i) == subz);
                            perf.(popNamesT{pop})(i) = data(idx).perf.(subPops{pop}).(['channel' num2str(chan)])(vv);
                            fr.(popNamesT{pop})(i) = data(idx).fr.(subPops{pop}).(['channel' num2str(chan)])(vv);
                        end
                    end

                    ax = subplot('Position', [xoffset yoffset + plotheight + 0.01 plotwidth plotheight * 0.2]);

                    %disp('Plotting the grids')
                    %disp([perf.(popNamesT{pop})])

                    plotPerfGrid([perf.(popNamesT{pop})], [fr.(popNamesT{pop})], '');
                    
                    %Double check to make sure this is not reversed
                    if popSizes(pop) == 1
                    %This will need to be changed if Ron this put back in.
                    approximate_grid = [perf.(popNamesT{pop});approximate_grid];
                    fr_grid = [fr.(popNamesT{pop});fr_grid];
                    end

                    if chan == 1, ylabel(subPops{pop}); end
                end

                % if we're plotting the population before the C cell (R2On),
                % add channel label titles
                if popSizes(pop) > 1, title(chanLabels{chan}, 'fontweight', 'normal'); end
            end
        end

        saveas(gcf, fullfile('simData', expName, ['C_grid_vary' num2str(vv) '.png']));
        %Might keep in just or memory considerations
        if plot_all == 0
            close all;
        else
            
        end
        %
    end
end
