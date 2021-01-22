% plotPerformanceGrids

%% performance grids
% performance vector has dimensions [numSpatialChan,nvaried]
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

% indices for target-masker configuraitons
fileNames = {ICfiles.name};
targetIdx = find(contains(fileNames,'m0') & contains({ICfiles.name},'_E')); %target only
maskerIdx = find(contains(fileNames,'s0') & contains({ICfiles.name},'_E')); %masker only
mixedIdx = find(~contains(fileNames,'m0') & ~contains(fileNames,'s0') & contains({ICfiles.name},'_E'));
% if there are no E vs I distinction
if isempty(targetIdx) && isempty(maskerIdx) && isempty(mixedIdx)
    targetIdx = find(contains(fileNames,'m0')); %target only
    maskerIdx = find(contains(fileNames,'s0')); %masker only
    mixedIdx = find(~contains(fileNames,'m0') & ~contains(fileNames,'s0'));
end

clear perf fr
for vv = 1:length(data(1).perf.C)
    h = figure('position',[200 200 600 600]);

    % C neuron; mixed cases
    if sum(ismember(subz,mixedIdx)) > 0
        for i = 1:length(mixedIdx)
            idx = (mixedIdx(i) == subz);
            perf.C(i) = data(idx).perf.C(vv);
            fr.C(i) = data(idx).fr.C(vv);
        end
        subplot('Position',[0.4 0.15 0.45 0.35])
        plotPerfGrid(perf.C',fr.C',[]);
    end

    % C neuron; target or masker only cases
    if sum(ismember(subz,targetIdx)) > 0
        perf.CT = zeros(1,4);
        perf.CM = zeros(1,4);
        fr.CT = zeros(1,4);
        fr.CM = zeros(1,4);
        if ~isempty(targetIdx)
            for i = 1:length(targetIdx)
                idx = (targetIdx(i) == subz);
                perf.CT(i) = data(idx).perf.C(vv);
                fr.CT(i) = data(idx).fr.C(vv);
            end
        end    
        if ~isempty(maskerIdx)
            for i = 1:length(maskerIdx)
                idx = (maskerIdx(i) == subz);
                perf.CM(i) = data(i).perf.C(vv);
                fr.CM(i) = data(i).fr.C(vv);
            end
        end    
        subplot('Position',[0.4 0.6 0.45 0.2])
        plotPerfGrid([perf.CT;perf.CM],[fr.CT;fr.CM],'Cortical');
    end
%     title([expVar ': ' num2str(varies(varied_param).range(vv))])
end
% % simulation info
% annotation('textbox',[.8 .85 .15 .2],...
%        'string',data(z).annot(vv,3:end),...
%        'FitBoxToText','on',...
%        'LineStyle','none')

% save grid
% Dirparts = strsplit(study_dir, filesep);
% DirPart = fullfile(Dirparts{1:end-1});
% saveas(gca,[DirPart filesep 'SpatialGrid vary ' variedParam num2str(varies(end).range(vv),'%0.2f') '.tiff'])
% clf
