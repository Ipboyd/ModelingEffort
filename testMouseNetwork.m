%% Tester Script that calls mouse_network function
clearvars

addpath('mechs')
addpath('dependencies')
addpath('eval_scripts')
addpath('genlib')
addpath(genpath('../dynasim'))

%% load IC spikes, convert cell array into binary array
y=1;
x=1;
ICdir = '104424';
ICdirPath = ['import' filesep ICdir filesep];
% ICdirPath = 'Z:\eng_research_hrc_binauralhearinglab\Model-Junzi_files_backup-remove_when_copied\V21\STRFs\163857\'
ICstruc = dir([ICdirPath '*.mat']); % Having issues here where it is not actually recieving the imported data
%fprintf("ICstruc length is %d\n",length(ICstruc));
performanceMax=zeros(1,16);
pMaxm0=zeros(1,4);
pMaxs0=zeros(1,4);
maxTaus0=zeros(1,4);
maxTaum0=zeros(1,4);
maxTau=zeros(1,16);
plot_rasters = 0;

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
%     study_dir = fullfile(pwd, 'run', mfilename, filesep, num2str(z));
    study_dir = fullfile(pwd, 'run', num2str(z));
%     if ~exist(study_dir,'dir')
%         mkdir(fullfile(study_dir,'solve'));
%     end
%     if exist(study_dir, 'dir')
%       rmdir(study_dir, 's');
%     end
    if exist(study_dir, 'dir')
      rmdir(study_dir, 's');
    end
    mkdir(fullfile(study_dir, 'solve'));
    save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');

    if ICstruc(z).name(2)=='0'
        [performance, tau]= mouse_network(study_dir,size(spks,3),plot_rasters);
        % Take the max of the performance
        pMaxs0(z)=max(performance);
        maxTaus0(z)=tau(max(performance)==performance);

    elseif ICstruc(z).name(4)=='0' %target only
        figure;
        for ii = 1:4
            subplot(1,4,ii)
            plotSpikeRasterFs(logical(squeeze(spks(:,ii,:))), 'PlotType','vertline');
            xlim([0 2000])
        end
        [performance, tau]= mouse_network(study_dir,size(spks,3),plot_rasters);
        % Take the max of the performance
        pMaxm0(y)=max(performance);
        maxTaum0(y)=tau(max(performance)==performance);
        y= y+1;
    else
        [performance, tau]= mouse_network(study_dir,size(spks,3),plot_rasters);
        % Take the max of the performance
        performanceMax(x)=max(performance);
        maxTau(x)=mean(tau(max(performance)==performance));
        x=x+1;
    end
end


%% output 1 by 4 grid of only masker
figure
l=num2cell(pMaxm0);
l= cellfun(@num2str, l,'UniformOutput', false);
positionVector = [0.1 0.25 0.85 0.1];
subplot('Position',positionVector)
im=imagesc(pMaxm0);
x = [1,2,3,4];
y=[1,1,1,1];
t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
for i= 1:4
    t(i).FontSize=12;
    if pMaxm0(i)>85
        t(i).Color = 'k';
    else
        t(i).Color= 'w';
    end
end
colormap('parula'); % this makes the boxes blue
colorbar;
caxis([50 100])
%% output 1 by 4 grid of only target
l=num2cell(pMaxs0);
l= cellfun(@num2str, l,'UniformOutput', false);
positionVector = [0.1 0.1 0.85 0.1];
subplot('Position',positionVector)
im=imagesc(pMaxs0);
x = [1,2,3,4];
y=[1,1,1,1];
t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
for i= 1:4
    t(i).FontSize=12;
    if pMaxs0(i)>85
        t(i).Color = 'k';
    else
        t(i).Color= 'w';
    end
end
colormap('parula'); % this makes the boxes blue
colorbar;
caxis([50 100])
%% output- 4 by 4 grid- calculate in a seperate script
% plot the matrix similar to the that on the paper- imagesc()
mat= reshape(performanceMax,4,4);
mat = flipud(mat);
l=num2cell(mat);
l= cellfun(@num2str, l,'UniformOutput', false);
positionVector = [0.275 0.45 0.5 0.5];
subplot('Position',positionVector)
im=imagesc(mat);
x = repmat(1:4,4,1);
y=x';
t= text(x(:),y(:), l, 'HorizontalAlignment', 'Center');
for i= 1:16
    t(i).FontSize=12;
    if mat(i)>85
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

%positionVector = [x0+subplotloc*(dx+lx) y0 lx ly];
%subplot('Position',positionVector)
