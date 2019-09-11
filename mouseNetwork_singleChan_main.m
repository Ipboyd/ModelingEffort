%% Main script for calling mouse data simulation network
% 2019-08-14 plotting now handles multiple varied parameters
clearvars;
close all;

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('../dynasim'))

ICdir = 'ICSimStim\mouse\v2\155210_seed142307_s30';
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
    if ICstruc(z).name(2)=='0' %masker only
        [pMaxs0(:,z,:),annotstr]= mouse_network_singleChan(study_dir,time_end,varies,plot_rasters,ICstruc(z).name);
    elseif ICstruc(z).name(4)=='0' %target only
        [pMaxm0(:,y,:),annotstr]= mouse_network_singleChan(study_dir,time_end,varies,plot_rasters,ICstruc(z).name);
        y= y+1;
    else
        [performanceMax(:,x,:),annotstr]= mouse_network_singleChan(study_dir,time_end,varies,plot_rasters,ICstruc(z).name);
        x=x+1;
    end
end

%         figure;
%         for ii = 1:4
%             subplot(1,4,ii)
%             plotSpikeRasterFs(flipud(logical(squeeze(spks(:,ii,:)))), 'PlotType','vertline');
%             xlim([0 2000])
%         end

%% performance grids
neurons = {'left sigmoid','gaussian','u','right sigmoid'};

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