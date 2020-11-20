%% Main script for calling mouse data simulation network
% 2019-08-14 plotting now handles multiple varied parameters
clearvars;
close all;

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('../dynasim'))

ICdir = 'ICSimStim\mouse\v2\145638_s30';
% ICdirPath = 'Z:\eng_research_hrc_binauralhearinglab\Model-Junzi_files_backup-remove_when_copied\V21\STRFs\163857\'
ICdirPath = [ICdir filesep];
ICstruc = dir([ICdirPath '*.mat']);

%% varied parameters
varies(1).conxn = '(IC->IC)';
varies(1).param = 'trial';
varies(1).range = 1:20;

varies(end+1).conxn = '(S->R)';
varies(end).param = 'gSYN';
varies(end).range = 0.15; %0.15:0.005:0.19;
% 
% varies(end+1).conxn = '(IC->R)';
% varies(end).param = 'gSYN';
% varies(end).range = [0.18,.2]; %0.15:0.005:0.19;
%% Initialize variables
plot_rasters = 1;

y=1;
x=1;
numSpatialChan = 4;
nIC = length(ICstruc);
nvaried = {varies(2:end).range};
nvaried = prod(cellfun(@length,nvaried));
performanceMax=zeros(numSpatialChan,16,nvaried);
pMaxm0=zeros(numSpatialChan,4,nvaried);
pMaxs0=zeros(numSpatialChan,4,nvaried);
diagConfigs = [6,12,18,24];
datetime=datestr(now,'yyyymmdd-HHMMSS');

data = struct();

for z = 1:length(ICstruc)
    % restructure IC spikes
    load([ICdirPath ICstruc(z).name],'t_spiketimes');
    temp = cellfun(@max,t_spiketimes,'UniformOutput',false);
    tmax = max([temp{:}]);
    spks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
    for j = 1:size(t_spiketimes,1) %trials [1:10]
        for k = 1:size(t_spiketimes,2) %neurons [(1:4),(1:4)]
            if k < 5 %song 1
                spks(j,k,round(t_spiketimes{j,k})) = 1;
            else
                spks(j+10,k-4,round(t_spiketimes{j,k})) = 1;
            end
        end
    end

    % save spk file
    spatialConfig = strsplit(ICstruc(z).name,'.');
    study_dir = fullfile(pwd, 'run', datetime, filesep, spatialConfig{1});
    if exist(study_dir, 'dir')
      rmdir(study_dir, 's');
    end
    mkdir(fullfile(study_dir, 'solve'));
    save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');

    % call network
    time_end = size(spks,3);
    [data(z).perf, data(z).annot] = mouse_network_singleChan(study_dir,time_end,varies,plot_rasters,ICstruc(z).name);
    data(z).name = ICstruc(z).name;
end

%         figure;
%         for ii = 1:4
%             subplot(1,4,ii)
%             plotSpikeRasterFs(flipud(logical(squeeze(spks(:,ii,:)))), 'PlotType','vertline');
%             xlim([0 2000])
%         end

%% performance grids
% performance vector has dimensions [numSpatialChan,nvaried]
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

temp = {data.name};
targetIdx = find(contains(temp,'m0'));
maskerIdx = find(contains(temp,'s0'));
mixedIdx = setdiff(1:length(data),[targetIdx,maskerIdx]);
[X,Y] = meshgrid(1:4,4:-1:1);
textColorThresh = 70;

for vv = 1:nvaried
    for i = 1:length(mixedIdx)
        perf(i,:) = data(mixedIdx(i)).perf(:,vv);
    end
    figure;
    for nn = 1:numSpatialChan
        subplot(2,2,nn)
        neuronPerf = perf(:,nn);
        str = cellstr(num2str(round(neuronPerf)));
        neuronPerf = reshape(neuronPerf,4,4);
        imagesc(flipud(neuronPerf));
        colormap('parula');
        xticks([1:4]); xticklabels({'-90','0','45','90'})
        yticks([1:4]); yticklabels(fliplr({'-90','0','45','90'}))
        title(neurons(nn))
        t = text(X(:),Y(:),str,'Fontsize',12,'HorizontalAlignment', 'Center');
        for i= 1:numel(neuronPerf)
            if neuronPerf(i)>textColorThresh
                t(i).Color = 'k';
            else
                t(i).Color= 'w';
            end
        end
        caxis([50,100])
        xlabel('Song Location')
        ylabel('Masker Location')
        set(gca,'fontsize',12)
    end
end
            
        
%{
for vv = 1:nvaried
    for nn = 1:numSpatialChan
    % output 1 by 4 grid of only masker
    figure
    textColorThresh = 70;
    l=num2cell(round(pMaxm0(nn,:,vv)));
    l= cellfun(@num2str, l,'UniformOutput', false);
    positionVector = [0.35 0.8 0.35 0.1];
    subplot('Position',positionVector)
    im=imagesc(pMaxm0(nn,:,vv));
    x = [1,2,3,4];
    y=[1,1,1,1];
    t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
    for i= 1:4
        t(i).FontSize=12;
        if pMaxm0(i,vv)>textColorThresh
            t(i).Color = 'k';
        else
            t(i).Color= 'w';
        end
    end
    colormap('parula'); % this makes the boxes blue
    caxis([50 100])
    % output 1 by 4 grid of only target
    l=num2cell(round(pMaxs0(nn,:,vv)));
    l= cellfun(@num2str, l,'UniformOutput', false);
    positionVector = [0.35 0.65 0.35 0.1];
    subplot('Position',positionVector)
    im=imagesc(pMaxs0(nn,:,vv));
    x = [1,2,3,4];
    y=[1,1,1,1];
    t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
    for i= 1:4
        t(i).FontSize=12;
        if pMaxs0(i,vv)>textColorThresh
            t(i).Color = 'k';
        else
            t(i).Color= 'w';
        end
    end
    colormap('parula'); % this makes the boxes blue
    caxis([50 100])
    % output- 4 by 4 grid- calculate in a seperate script
    % plot the matrix similar to the that on the paper- imagesc()
    mat = reshape(performanceMax(nn,:,vv),4,4);
    mat = flipud(mat);
    l = num2cell(round(mat));
    l = cellfun(@num2str, l,'UniformOutput', false);
    positionVector = [0.35 0.15 0.4 0.4];
    subplot('Position',positionVector)
    im=imagesc(mat);
    x = repmat(1:4,4,1);
    y=x';
    t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
    for i= 1:16
        t(i).FontSize=12;
        if mat(i)>textColorThresh
            t(i).Color = 'k';
        else
            t(i).Color= 'w';
        end
    end
    colormap('parula'); % this makes the boxes blue
    colorbar;
    caxis([50 100])
    xlabel('song location')
    ylabel('masker location')
    yticks(1:4)
    yticklabels({'90','45','0','-90'})
    xticks(1:4)
    xticklabels({'-90','0','45','90'})
    set(gca,'fontsize',12)
    %positionVector = [x0+subplotloc*(dx+lx) y0 lx ly];
    %subplot('Position',positionVector)
    
    % save grid
    ICparts = strsplit(ICdir, filesep);
    suptitle(neurons{nn})
    annotation('textbox',[.75 .6 .2 .1],...
               'string',annotstr(vv,:),...
               'FitBoxToText','on',...
               'LineStyle','none')

    Dirparts = strsplit(study_dir, filesep);
    DirPart = fullfile(Dirparts{1:end-1});
    saveas(gca,[DirPart filesep 'performance_grid_v ' num2str(vv) '.tiff'])
    end
end
%}