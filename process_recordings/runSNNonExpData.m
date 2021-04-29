% generate training data from custom-designed AIM network for optimizing
% network weights with matlab's DNN toolbox.
%
% inputs: user specified, raw IC output
% network structure: user specified

cd('C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid')
dynasimPath = 'C:\Users\Kenny\Desktop\GitHub\DynaSim';
ICdir = 'C:\Users\Kenny\Desktop\GitHub\MouseSpatialGrid\data_recordings\reconfigured';
ICfile = '616283 6TMR 2dB chan27 noise3 gain100.mat';
addpath('mechs')
addpath('genlib')
addpath('plotting')
addpath(genpath(dynasimPath))
expName = '616283_6TMR_2dB_chan27 001';

debug_flag = 0;
save_flag = 0;

% setup directory for current simulation
datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);
if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));
simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

% % % get indices of STRFS, all locations, excitatory inputs only
% % ICfiles = dir([ICdir filesep '*.mat']);
% % subz = 1:length(ICfiles);
% % % subz = [[20:-5:5],fliplr([6:9,11:14,16:19,21:24])]; %to match experiment data
% % % subz = [20:-5:5];
% % % subz = find(~contains({ICfiles.name},'s0')); % exclude masker-only.
% % % subz = find(contains({ICfiles.name},'s1m2'));
% % % subz = [1:4,5,10,15,20,6,12,18,24]; %single channel
% % % subz = [5,7,10,11]; %channels 1 & 2
% % fprintf('found %i files matching subz criteria\n',length(subz));
% % 
% % % check IC inputs
% % if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

%% define network parameters
clear varies

dt = 0.1; %ms

gsyn_same = 0.18;

% custom parameters
varies(1).conxn = '(Inh->Inh,Exc->Exc)';
varies(1).param = 'trial';
varies(1).range = 1:20;

% deactivate TD neuron
varies(end+1).conxn = 'TD';
varies(end).param = 'Itonic';
varies(end).range = 0;

% inh neuron = sharpen
varies(end+1).conxn = 'Inh->Inh';
varies(end).param = 'g_postIC';
varies(end).range = 0.4;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'gSYN';
varies(end).range = 0;

varies(end+1).conxn = 'Inh->R';
varies(end).param = 'delay';
varies(end).range = 4;

varies(end+1).conxn = 'Inh';
varies(end).param = 'noise';
varies(end).range = 0;

varies(end+1).conxn = 'Exc->Exc';
varies(end).param = 'g_postIC';
varies(end).range = 0.9;

varies(end+1).conxn = 'Exc->R';
varies(end).param = 'gSYN';
varies(end).range = gsyn_same;

varies(end+1).conxn = 'Exc->X';
varies(end).param = 'gSYN';
varies(end).range = gsyn_same;

varies(end+1).conxn = 'R';
varies(end).param = 'noise';
varies(end).range = 0.1:0.1:0.4;

varies(end+1).conxn = 'X';
varies(end).param = 'noise';
varies(end).range = 0.5;

varies(end+1).conxn = 'X';
varies(end).param = 'Itonic';
varies(end).range = 0.18;

varies(end+1).conxn = 'X->R';
varies(end).param = 'gSYN';
varies(end).range = 0.35;

varies(end+1).conxn = 'C';
varies(end).param = 'noise';
varies(end).range = 0;

% R-C weights 0.18 by default
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN1';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN2';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN3';
varies(end).range = gsyn_same;
varies(end+1).conxn = 'R->C';
varies(end).param = 'gSYN4';
varies(end).range = gsyn_same;

% display parameters
network_params = [{varies.conxn}' {varies.param}' {varies.range}']

% find varied parameter other, than the trials
varied_param = find(cellfun(@length,{varies.range})>1);
if length(varied_param) > 1
    varied_param = varied_param(2); 
else
    varied_param = 2;
end
expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');
numVaried = length(varies(varied_param).range);

% specify netcons
netcons.xrNetcon = zeros(3); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(2,3) = 1;
% netcons.xrNetcon(1,4) = 1;
% netcons.xrNetcon(4,1) = 1;
% netcons.xrNetcon(4,2) = 1;
% netcons.xrNetcon(4,3) = 1;
% netcons.xrNetcon(4,2) = 1;
netcons.rcNetcon = [1 1 1]';

netcons.irNetcon = eye(3); %inh -> R; sharpening
netcons.tdxNetcon = zeros(3); % I2 -> I
netcons.tdrNetcon = zeros(3); % I2 -> R
%% prep input data
% concatenate spike-time matrices, save to study dir

load([ICdir filesep ICfile])
padToTime = 3200; %ms
label = {'E','I'};
for ICtype = [0,1] %only E no I
    spks = [];
    songloc = 0;
    maskloc = 0;
    z = 1;
    for songloc = 1:4
        singleConfigSpks = ICspks_clean(songloc).spks;

        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize)); 
        end
        % concatenate
        spks = cat(3,spks,singleConfigSpks);

        trialStartTimes(z) = padToTime;
        configOrder{z} = sprintf('s%im%i',songloc,maskloc);
        z = z+1;
    end
    for songloc = 1:4
        for maskloc = 1:4
            singleConfigSpks = ICspks_mixed(songloc,maskloc).spks;

            % pad each trial to have duration of timePerTrial
            if size(singleConfigSpks,3) < padToTime/dt
                padSize = padToTime/dt-size(singleConfigSpks,3);
                singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize)); 
            end
            % concatenate
            spks = cat(3,spks,singleConfigSpks);

            trialStartTimes(z) = padToTime;
            configOrder{z} = sprintf('s%im%i',songloc,maskloc);
            z = z+1;
        end
    end
%     spks = zeros(20,4,32000);
    spks(:,3,:) = [];
    save(fullfile(study_dir, 'solve',sprintf('IC_spks_%s.mat',label{ICtype+1})),'spks');
end

%% run simulation
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_2020');
options.plotRasters = 0;
options.time_end = size(spks,3)*dt; %ms;
options.locNum = [];
options.parfor_flag = 1;
[snn_out,s] = birdNetwork(study_dir,varies,netcons,options);

%% post process

% calculate performance
data = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
options.plotRasters = 0;
options.subPops = {'Exc','R','C'}; %individually specify population performances to plot
% options.subPops = {'X'};
configName = configOrder(1:20);
% configName = {'n/a'};
options.variedField = strrep(expVar,'-','_');
tic
for z = 1:length(configName)
    trialStart = PPtrialStartTimes(z);
    trialEnd = PPtrialEndTimes(z);
    figName = [simDataDir filesep configName{z}];
    [data(z).perf,data(z).fr,data(z).spks] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    
%     [dataOld(z).perf,dataOld(z).fr] = postProcessData(snn_out,trialStart,trialEnd,options);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if z == 1
    plot(varies(varied_param).range,data.perf.R.channel2)
    xlabel(expVar)
    ylabel('Perf')
    return;
end
%% plot rasters
locLabels = [90 45 0 -90];
chanLabels = {'ipsi sigmoid', 'gauss', 'contra sigmoid'};

subPops = {'R'};
popSizes = [snn_out(1).model.specification.populations.size];
popNames = {s.populations.name};
myPopSizes = popSizes(ismember(popNames,subPops));
figure;
for v = 1:numVaried
    for subPop = subPops
        for channel = 1:popSizes(ismember(popNames,subPop))
            figure('position',[100 100 900 800])
            for z = 1:4
                currentSpks = data(z).spks.(subPop{1})(v).(['channel' num2str(channel)]);
                subplot('position',[0.075+0.22*(z-1) 0.7 0.2 0.175])
                plotSpikeRasterFs(logical(currentSpks),'PlotType','vertline2');
                xticks([])
                yticks([])
                xlim([0 3000/dt])
                title(['performance: ' num2str(data(z).perf.(subPop{1}).(['channel' num2str(channel)])(v))])
            end

            for songloc = 1:4
                for maskloc = 1:4
%                     disp(configName{z}) % this line helps with debugging
                    z = 4 + (songloc-1)*4 + maskloc;
                    currentSpks = data(z).spks.(subPop{1})(v).(['channel' num2str(channel)]);
                    subplot('position',[0.075+0.22*(songloc-1) 0.075+0.15*(maskloc-1) 0.2 0.125])
                    plotSpikeRasterFs(logical(currentSpks),'PlotType','vertline2');
                    xticks([])
                    yticks([])
                    xlim([0 3000/dt])
                title(['performance: ' num2str(data(z).perf.(subPop{1}).(['channel' num2str(channel)])(v))])

                    if maskloc == 1
                        xlabel(['songloc' num2str(locLabels(songloc))])
                    end
                    if songloc == 1
                        ylabel(['maskloc' num2str(locLabels(maskloc))])
                    end
                end
            end
            if strcmp(subPop,'C')
                sgtitle({['Neuron: ', subPop{1}], [expVar, num2str(varies(varied_param).range(v))]})
            else
                sgtitle({['Neuron: ', subPop{1}], ['Channel: ', chanLabels{channel}], [expVar, num2str(varies(varied_param).range(v))]})
            end
        end
    end
end
%% plot performance
subPops = {'R'};
targetIdx = 1:4;
mixedIdx = 5:20;
simOptions.varied_param = varied_param;
simOptions.varies = varies;
simOptions.expVar = expVar;
simOptions.subz = 1:20; % configurations
simOptions.locationLabels = locLabels;
simOptions.chanLabels = chanLabels;
plotPerformanceGrids_v3(data,s,subPops,targetIdx,mixedIdx,simOptions)

% nSubPops = length(subPops);
% subplot(nSubPops,3,2); imagesc(netcons.xrNetcon);
% colorbar; caxis([0 1])
% 
% subplot(nSubPops,3,3); imagesc(netcons.rcNetcon);
% colorbar; caxis([0 1])

%% performance vs varied param
addpath('C:\Users\Kenny\Desktop\GitHub\Matlab Plotting Toolboxes\BrewerMap')
colors = brewermap(4,'set1')
songlocs = 1:4
masklocs = 0:4
lwidths = 1:0.2:2;
figure
for maskloc = masklocs
    subplot(5,1,maskloc+1);
    if maskloc == 0
        for songloc = songlocs
            z = songloc;
            perf = data(z).perf.C.channel1;
            plot(varies(varied_param).range,perf,'-','color',colors(songloc,:),'linewidth',lwidths(maskloc+1));
            hold on;
        end
    else
        for songloc = songlocs
            z = songloc*4+maskloc
            perf = data(z).perf.C.channel1;
            plot(varies(varied_param).range,perf,'-','color',colors(songloc,:),'linewidth',lwidths(maskloc+1));
            hold on;
        end
    end
%     ylim([50 100])
end
xlabel(expVar)
ylabel('performance')
legend('songloc 90','songloc 45','songloc 0','songloc -90')

return;
%%

%%%%%%%%%%%%%%%%%%%%%% save results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if save_flag
    perf_data.perf = data.perf;
    perf_data.fr = data.fr
    name = 'vary sharpening weight - xtonic1.4';
    save(['simdata' filesep name '.mat'],...
        'perf_data','netcons','network_params','options','s','ICdir','varies');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end

%%
% Rspks = snn_spks(variation).R.raw;
% Xspks = snn_spks(variation).X.raw;
% Espks = snn_spks(variation).E.raw;
% 
% figure;
% plotSpikeRasterFs(logical(Xspks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('X spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(Rspks(:,:,1)),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('R spike')
% xlim([2500 3200])
% 
% figure;
% plotSpikeRasterFs(logical(squeeze(spks(:,1,:))),'PlotType','vertline2','Fs',1/dt);
% xlim([0 snn_out(1).time(end)/dt])
% title('IC spike')
% xlim([2500 3200])

% figure;
% scaleFactor = 30;
% for trialToPlot = 1:20
%     plot(snn_out(2).time,snn_out(trialToPlot).R_V(:,1) + scaleFactor*(trialToPlot-1),'color', [0, 0.4470, 0.7410]); hold on;
%     plot(snn_out(2).time,snn_out(trialToPlot).X_V(:,2) + scaleFactor*(trialToPlot-1),'color', [0.8500, 0.3250, 0.0980]);
% end
% legend('R','X')
% xlabel('time')
% ylabel('V')
% ylim([-80 scaleFactor*20-70])
% yticks([-50:scaleFactor:scaleFactor*20-50])
% yticklabels([1:scaleFactor])