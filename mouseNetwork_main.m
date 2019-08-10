%% Tester Script that calls mouse_network function
clearvars;
close all;

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('../dynasim'))

% to do: re-organize loop to handle outputs form multiple V2s


%% load IC spikes, convert cell array into binary array
y=1;
x=1;
ICdir = 'ICSimStim\bird\FRgain1\163516_seed142307_s20';
ICdirPath = [ICdir filesep];
% ICdirPath = 'Z:\eng_research_hrc_binauralhearinglab\Model-Junzi_files_backup-remove_when_copied\V21\STRFs\163857\'
ICstruc = dir([ICdirPath '*.mat']);

plot_rasters = 1;

performanceMax=zeros(1,16);
pMaxm0=zeros(1,4);
pMaxs0=zeros(1,4);
maxTaus0=zeros(1,4);
maxTaum0=zeros(1,4);
maxTau=zeros(1,16);
diagConfigs = [6,12,18,24];
datetime=datestr(now,'yyyymmdd-HHMMSS');
for z = 1:length(ICstruc)
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
        [pmaxs0(z,:), maxTaus0(z,:),v2]= mouse_network(study_dir,time_end,plot_rasters,ICstruc(z).name);
    elseif ICstruc(z).name(4)=='0' %target only
        [pMaxm0(y,:), maxTaum0(y,:),v2]= mouse_network(study_dir,time_end,plot_rasters,ICstruc(z).name);
        y= y+1;
    else
        [performanceMax(x,:),maxTau(x,:),v2]= mouse_network(study_dir,time_end,plot_rasters,ICstruc(z).name);
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
for vv = 1:length(v2)
    % output 1 by 4 grid of only masker
    figure
    textColorThresh = 70;
    l=num2cell(round(pMaxm0));
    l= cellfun(@num2str, l,'UniformOutput', false);
    positionVector = [0.35 0.8 0.35 0.1];
    subplot('Position',positionVector)
    im=imagesc(pMaxm0);
    x = [1,2,3,4];
    y=[1,1,1,1];
    t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
    for i= 1:4
        t(i).FontSize=12;
        if pMaxm0(i)>textColorThresh
            t(i).Color = 'k';
        else
            t(i).Color= 'w';
        end
    end
    colormap('parula'); % this makes the boxes blue
    caxis([50 100])
    % output 1 by 4 grid of only target
    l=num2cell(round(pMaxs0));
    l= cellfun(@num2str, l,'UniformOutput', false);
    positionVector = [0.35 0.65 0.35 0.1];
    subplot('Position',positionVector)
    im=imagesc(pMaxs0);
    x = [1,2,3,4];
    y=[1,1,1,1];
    t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
    for i= 1:4
        t(i).FontSize=12;
        if pMaxs0(i)>textColorThresh
            t(i).Color = 'k';
        else
            t(i).Color= 'w';
        end
    end
    colormap('parula'); % this makes the boxes blue
    caxis([50 100])
    % output- 4 by 4 grid- calculate in a seperate script
    % plot the matrix similar to the that on the paper- imagesc()
    mat = reshape(performanceMax,4,4);
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
    suptitle({['data: ' ICparts{end}],['v2 = ' num2str(v2(vv))]})

    Dirparts = strsplit(study_dir, filesep);
    DirPart = fullfile(Dirparts{1:end-1});
    saveas(gca,[DirPart filesep 'performance_grid.tiff'])
end