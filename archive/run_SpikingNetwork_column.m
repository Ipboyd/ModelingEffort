% inputs: user specified, raw IC output
% network structure: user specified

cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid')
dynasimPath = '../DynaSim';

% absolute refractory period of 1ms w/ relative refractory period of 1ms,
% steepeness recovery factor of 2
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\vary_recovery\alpha_0.010\STRFgain 0.10, 1.0ms tau_rel';

% sharper tuning curves
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\sharpened_curves//alpha_0.010//STRFgain 0.10, 1.0ms tau_rel';


% ushaped chanel is now contralateral sigmoid that inhibits center channel
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\inhib-input-to-center//alpha_0.010//STRFgain 0.10, 1.0ms tau_rel'
% ICdir =    'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\inhib-input-to-center-V2//alpha_0.010//STRFgain 0.10, 1.0ms tau_rel'
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\inhib-input-to-center-V3//alpha_0.010//STRFgain 0.10, 1.0ms tau_rel';

ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid\ICSimStim\mouse\best_curves//alpha_0.010//STRFgain-0.10-1.0ms-tau_rel';

addpath('mechs'); addpath('fixed-stimuli');
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');

expName = '06-11-2022 adaptation curves';

model = struct; model.type = '';
model.interaction = 0;

options =struct;
options.mex_flag = 0;

% debug_flag = 1; save_flag = 0;

study_dir = fullfile(pwd,'run','m-file-final');
%if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
%mkdir(fullfile(study_dir, 'solve'));
simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

% get indices of STRFS, all locations, excitatory inputs only
ICfiles = dir([ICdir filesep '*.mat']);
subz = 1:length(ICfiles);
% fprintf('found %i files matching subz criteria /n',length(subz));

% check IC inputs
if ~exist(ICdir,'dir'), restructureICspks(ICdir); end

%% define network parameters
clear varies

options.nCells = 3;
dt = 0.1; %ms

nSims = 1;
trialInds = repmat(1:20,nSims,1);

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = 'IC->IC';
varies(1).param = 'trial';
varies(1).range = trialInds(:)';
% % % DO NOT CHANGE THIS % % %
 
% within-channel inhibition
varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = 'gSYN';
varies(end).range = 0.02; %[ 0 : 0.005 : 0.04 ];

% recurrent excitation of PV cells
varies(end+1).conxn = '(R1->S1,R2->S2)';
varies(end).param = 'gSYN';
varies(end).range = 0.02; % [ 0 : 0.005 : 0.04 ] ;

% % feed-forward excitation of PV cells 
% varies(end+1).conxn = '(IC->S1,R1->S2)';
% varies(end).param = '(tauP,fP,gSYN)';
% varies(end).range = [ 80 ; 0.2 ; 0.02 ];

% control and opto conditions
varies(end+1).conxn = '(S1,S2)';
varies(end).param = 'Itonic';
varies(end).range = 0; 

% cross-channel inhibition - inputs to SOM neurons
varies(end+1).conxn = '(R1->X1,R2->X2)';
varies(end).param = '(tauF,fF,gSYN)';
varies(end).range =  [ 120 ; 0.2 ; 0.0025 ]; 

% cross-channel inhibition - SOM outputs
varies(end+1).conxn = '(X1->R1,X2->R2)';
varies(end).param = '(tauF,fF)';
varies(end).range =  [ 120 ; 0.1 ]; 

% full grid, with and without cross channel
varies(end+1).conxn = '(X1->R1,X2->R2)';
varies(end).param = 'gSYN';
varies(end).range =  [ 0.0025 ]; 

varies(end+1).conxn = '(R2->R2,C->C)';
varies(end).param = '(FR,sigma)';
varies(end).range = [ 0 ; 0 ]; % 8 12 ; 6 6

% adaptation recovery times
varies(end+1).conxn = '(R1,R2,C)';
varies(end).param = 'tau_ad';
varies(end).range = [ 1 60 120 180 ]; % 8 12 ; 6 6

% find varied parameter, excluding trials
varied_param = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));

if length(varied_param) > 1
    varied_param = varied_param(2); 
else % if no varied params, settle on 2nd entry in varies
    varied_param = 2;
end

% netcons can't be put in the varies struct (Dynasim doesn't recognize it?);
% it needs to be put in a different struct

netcons = struct; % row = source, column = target
netcons.XRnetcon = zeros(options.nCells,options.nCells);
if options.nCells == 2
    netcons.XRnetcon(2,1) = 1; % 0deg channel inhibits 90deg
elseif options.nCells == 3
    netcons.XRnetcon([2 2],[1 3]) = 1; % 0deg channel inhibits others
elseif options.nCells == 4
    netcons.XRnetcon([3 3],[1 4]) = 1; % 0deg channel inhibits others
end

expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');
numVaried = length(varies(varied_param).range);

%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz)); %ms
padToTime = 3500; %ms
label = {'On'};

for ICtype = 0 % only E no I
    % divide all times by dt to upsample the time axis
    spks = [];
    for z = 1:length(subz)
        % disp(ICfiles(subz(z)+0).name); %read in E spikes only
        load([ICdir filesep ICfiles(subz(z)).name],'t_spiketimes_on');
        
        % convert spike times to spike trains. This method results in
        % dt = 1 ms
        temp = cellfun(@max,t_spiketimes_on,'UniformOutput',false);
        tmax = round(max([temp{:}])/dt);
        singleConfigSpks = zeros(20,4,tmax); %I'm storing spikes in a slightly different way...
        for j = 1:size(t_spiketimes_on,1) %trials [1:10]
            for k = 1:size(t_spiketimes_on,2) %neurons [(1:4),(1:4)]
                if k < 5 %song 1
                    if ICtype == 0
                        singleConfigSpks(j,k,round(t_spiketimes_on{j,k}/dt)) = 1;
                    else
                        singleConfigSpks(j,k,round(t_spiketimes_off{j,k}/dt)) = 1;
                    end
                    
                else
                    if ICtype == 0
                        singleConfigSpks(j+10,k-4,round(t_spiketimes_on{j,k}/dt)) = 1;
                    else
                        singleConfigSpks(j+10,k-4,round(t_spiketimes_off{j,k}/dt)) = 1;
                    end
                end
            end
        end
        % singleConfigSpks(:,2,:) = 0; % zero out the U channel
        
        trialStartTimes(z) = padToTime;
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize));
        end
        
        % for single channel, since column network only has one location
        if options.nCells == 1, spks = cat(3,spks,singleConfigSpks(:,3,:));
        elseif options.nCells == 2, spks = cat(3,spks,singleConfigSpks(:,[1 3],:));
        elseif options.nCells == 3, spks = cat(3,spks,singleConfigSpks(:,[1 3 4],:));
        elseif options.nCells == 4, spks = cat(3,spks,singleConfigSpks);
        end
        
    end

    spks(isnan(spks)) = 0;
    save(fullfile(study_dir, 'solve','IC_spks.mat'),'spks');
end

%% run simulation
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_202');
options.plotRasters = 0;

if options.nCells == 1, options.locNum = 15; % 15 = clean target at 0deg
else, options.locNum = 16; % 16 = target at 0deg, masker at 90deg
end
% options.locNum = 18; % colocated at 0deg
% options.locNum = 8; % target at 90deg, masker at 0deg
options.locNum = 19;
options.time_end = padToTime;

% % all locations
% options.time_end = size(spks,3)*dt; %ms;
% options.locNum = [];

options.parfor_flag = 0;

[snn_out,s] = columnNetwork(study_dir,varies,options,netcons);
% [snn_out,s] = columnNetwork_Wehr(study_dir,varies,options,netcons);

%% post-process

% figure; 
% plotsimPSTH('',snn_out); plotsimPSTH('R1',snn_out); plotsimPSTH('R2',snn_out);
% legend('','R1','R2');

% if numVaried <= 5
%     for n = 1:numVaried
%         plotUnitVoltage('C',snn_out,n);
%         saveas(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '.png']);
%         savefig(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '']);
%         
%         plotUnitVoltage('C',snn_out,n);
%         saveas(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '.png']);
%         savefig(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '']);
%         
%         plotUnitVoltage('C',snn_out,n);
%         saveas(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '.png']);
%         savefig(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '']);
%         
%         plotUnitVoltage('C',snn_out,n);
%         saveas(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '.png']);
%         savefig(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '']);
%     end
% end

% calculate performance
data = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
configName = cellfun(@(x) strsplit(x,'_'),{ICfiles(subz).name}','UniformOutput',false);
configName = vertcat(configName{:}); configName = configName(:,1);
options.variedField = strrep(expVar,'-','_');
options.chansToPlot = [1 2];

annotTable = createSimNotes(snn_out,expName,model,options);

% save C spikes and varied params to struct
names = snn_out(1).varied; results = struct;
for i = 1:length(snn_out)
results(i).C_V_spikes = snn_out(i).C_V_spikes;

for j = 1:length(names)
results(i).(names{j}) = snn_out(i).(names{j});
end
end
results(1).model = snn_out(1).model;
save([simDataDir filesep 'C_results.mat'],'results');


% if length(snn_out)/20 == numVaried
tic
if ~isempty(options.locNum)
    trialStart = 1; trialEnd = padToTime/dt;
    figName = [simDataDir filesep configName{options.locNum}(1:end-4)];
    [data.perf,data.fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options,model);
else
    for z = 1:length(subz)
        trialStart = PPtrialStartTimes(z);
        trialEnd = PPtrialEndTimes(z);
        figName = [simDataDir filesep configName{z}];
        [data(z).perf,data(z).fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
        plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options,model);
    end
end
toc
% close all;
 
if nSims == 5
    [pc,fr]= plotParamvsPerf_1D(results,numVaried);

    pc_trials = struct2cell(pc);

    ctrl_mean = cellfun(@(x) mean(x(:,1)),pc_trials);
    laser_mean = cellfun(@(x) mean(x(:,2)),pc_trials);

    ctrl_se = cellfun(@(x) std(x(:,1))/sqrt(numel(x(:,1))),pc_trials);
    laser_se = cellfun(@(x) std(x(:,2))/sqrt(numel(x(:,2))),pc_trials);

    figure('unit','inches','position',[5 5 3 3]);
    bar((1:4)-.2,ctrl_mean,0.4,'facecolor','none','linewidth',2); hold on;
    bar((1:4)+.2,laser_mean,0.4,'facecolor','k','linewidth',2);
    xlim([0.4 4.6]);

    errorbar((1:4)-.2,ctrl_mean,ctrl_se,'color','k','linestyle','none','linewidth',1); hold on;
    errorbar((1:4)+.2,laser_mean,laser_se,'color','k','linestyle','none','linewidth',1);

    ylim([50 100]);
    set(gca,'xticklabels',{'SPIKE','ISI','RI-SPIKE','Spike count'},'xtick',1:4,'fontsize',8);
    ytickformat('percentage');
    ylabel('Performance'); legend('Control','Laser');
    saveas(gcf,[simDataDir filesep 'opto_performance_results.fig']);
end

% grid search plots
if numVaried >= 10
    names = {s.populations.name};
    dontPlot = {'IC'};
    names(matches(names,dontPlot)) = [];
    
    for p = 1:length(names)
        plotPerfvsParams(names{p},data,varies,simDataDir)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results

% close all
% 
% figure; vizNetwork(s,0,'C','')
% saveas(figure(1),fullfile('simData',expName,'netcons.png'));
% saveas(figure(2),fullfile('simData',expName,'network.png'));
% close all;
% 

if isempty(options.locNum)
    targetIdx = find(cellfun(@(x) contains(x,'m0'),{ICfiles.name}));
    mixedIdx = find(cellfun(@(x) ~contains(x,'m0') && ~contains(x,'s0'),{ICfiles.name}));

    simOptions = struct;
    
    simOptions.subz = subz;
    simOptions.varies = varies;
    simOptions.varied_param = varied_param;
    simOptions.locationLabels = {'90','45','0','-90'};
    simOptions.expVar = expVar;
    simOptions.chanLabels = simOptions.locationLabels;
    
    options.subPops = {'R2','C'};
    plotPerformanceGrids_v3(data,s,annotTable,options.subPops,targetIdx,mixedIdx,simOptions,expName);
    
end


