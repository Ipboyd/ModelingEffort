cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid')
dynasimPath = '../DynaSim';

addpath('mechs'); addpath('fixed-stimuli'); addpath(genpath('ICSimStim'));
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');

expName = '06-22-2022 MGB, talk PSTH';

options = struct;
options.nCells = 1;

options.mex_flag = 0;
options.parfor_flag = 0;
options.STRFgain = 0.13;
options.plotRasters = 0;

% 1 for spatial attention, 0 for passive condition
options.SpatialAttention = 0;
options.Imask = [0 1 0];

% locNum should be empty for full grids
options.locNum = 15;

% debug_flag = 1; save_flag = 0;

if options.SpatialAttention
study_dir = fullfile(pwd,'run','attention-V2');
else
study_dir = fullfile(pwd,'run','m-file-final');
end
if options.nCells == 1, study_dir = fullfile(pwd,'run','single-channel'); end

% if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
% mkdir(fullfile(study_dir, 'solve'));

simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

%% define network parameters
clear varies

dt = 0.1; %ms

nSims = 1;
trialInds = repmat(1:20,nSims,1);

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = '(ICE->ICE,ICI->ICI)';
varies(1).param = 'trial';
varies(1).range = trialInds(:)';
% % % DO NOT CHANGE THIS % % %
 
% within-channel inhibition
varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = 'gSYN';
varies(end).range = [ 0.02 ];

varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = 'tauP';
varies(end).range = [ 120 ];

varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = 'fP';
varies(end).range = [ 0.5 ];

% recurrent excitation of PV cells
varies(end+1).conxn = '(R1->S1,R2->S2)';
varies(end).param = 'gSYN';
varies(end).range = [ 0.02 ];

% % feed-forward excitation of PV cells 
% varies(end+1).conxn = '(IC->S1,R1->S2)';
% varies(end).param = '(tauP,fP,gSYN)';
% varies(end).range = [ 80 ; 0.2 ; 0.02 ];

% control and opto conditions
varies(end+1).conxn = '(S1,S2)';
varies(end).param = 'Itonic';
varies(end).range = 0; 

% % cross-channel inhibition - inputs to SOM neurons
% varies(end+1).conxn = '(R1->X1,R2->X2)';
% varies(end).param = '(tauF,fF,gSYN)';
% varies(end).range =  [ 120 ; 0.2 ; 0.0025 ]; 
% 
% % cross-channel inhibition - SOM outputs
% varies(end+1).conxn = '(X1->R1,X2->R2)';
% varies(end).param = '(tauF,fF,gSYN)';
% varies(end).range =  [ 120 ; 0 ; 0.002 ];  %0.002

varies(end+1).conxn = '(R2->R2,C->C)';
varies(end).param = '(FR,sigma)';
varies(end).range = [ 5 ; 0 ]; % 8 12 ; 6 6

% adaptation recovery times
varies(end+1).conxn = '(R1,R2,C)';
varies(end).param = '(g_ad_inc,tau_ad)';
varies(end).range = [ 0.0001 ; 60 ];

% find varied parameter, excluding trials
varied_param = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));
varied_param(1) = []; % delete trial varies

if isempty(varied_param) % if no varied params, settle on 2nd entry in varies
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
% elseif options.nCells == 4
%     netcons.XRnetcon([3 3],[1 4]) = 1; % 0deg channel inhibits others
end

% for simplification: use 1st varied param for 2d searches
expVar = [varies(varied_param(1)).conxn '-' varies(varied_param(1)).param];
expVar = strrep(expVar,'->','_');

% calculate # parameter sets
numVaried = prod(cellfun(@length,{varies(varied_param).range}));

%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,24); %ms
padToTime = 3500; %ms
label = {'On'};

locs = [90 45 0 -90];

% z : 1-4 , masker only
% z : 5,10,15,20 , target only
% z : 6-9 and etc. , mixed trials

for z = 1:24
    trialStartTimes(z) = padToTime;
end

load('FR_traces_search.mat');
temp = cellfun(@numel,fr_target);
tmax = max(temp);

x = -108:108;

% excitatory tuning curves
tuningcurve_E(1,:) = 0.8*sigmf(x,[0.1 10])/sigmf(90,[0.1 10]) + 0.2; % = 1 at x = +90
tuningcurve_E(2,:) = 1 - 0.8*gaussmf(x,[38 0]);
tuningcurve_E(3,:) = 0.8*gaussmf(x,[24 0]) + 0.2;
tuningcurve_E(4,:) = 0.8*sigmf(x,[-0.1 -10])/sigmf(-90,[-0.1 -10]) + 0.2; % at -90°, sigmoid == 1

% inhibitory tuning curves
tuningcurve_I(1,:) = 0.8*sigmf(x,[0.1 0])/sigmf(90,[0.1 0]) + 0.2; % = 1 at x = +90
tuningcurve_I(2,:) = 1 - 0.8*gaussmf(x,[38 0]);
tuningcurve_I(3,:) = 0.63*gaussmf(x,[45 0]);
tuningcurve_I(4,:) = 0.8*sigmf(x,[-0.1 0])/sigmf(-90,[-0.1 0]) + 0.2; % at -90°, sigmoid == 1

if options.nCells == 3
    figure;
    plot(x,tuningcurve_E([1 3 4],:)','linewidth',1);
    hold on; plot(x,sum(tuningcurve_E([1 3 4],:)),'k','linewidth',2);
    legend('Contra','Center','Ipsi'); title('Excitatory tuning curves');
    xlim([-108 108]); ylim([0 2]); set(gca,'xtick',[-90 0 45 90],'xdir','reverse');
    saveas(gcf,[simDataDir filesep 'excitatory tuning curves.png']);

    figure;
    plot(x,tuningcurve_I([1 3 4],:)','linewidth',1);
    hold on; plot(x,sum(tuningcurve_I([1 3 4],:)),'k','linewidth',2);
    legend('Contra','Center','Ipsi'); title('PV tuning curves');
    xlim([-108 108]); ylim([0 2]); set(gca,'xtick',[-90 0 45 90],'xdir','reverse');
   saveas(gcf,[simDataDir filesep 'inhibitory tuning curves.png']);
end
close all;

% only create spks file if not done yet
if ~exist([study_dir filesep 'solve' filesep 'IC_spks_E.mat'],'file')
    labels = {'E','I'};
for ICtype = [1 2] % only E no I
    % divide all times by dt to upsample the time axis
    spks = [];
        
    % load fr traces, scale by weights
    % dt = 0.1 ms

    for z = 1:24

        % target and masker weights
        if z <= 4  % masker only
            tloc(z) = nan;
            mloc(z) = locs(z);
        elseif mod(z,5) == 0 % target only
            tloc(z) = locs(floor(z/5));
            mloc(z) = nan;
        else % mixed
            tloc(z) = locs(floor(z/5));
            mloc(z) = locs(mod(z,5));
        end

        singleConfigSpks = zeros(20,4,tmax);
        
        for t = 1:20 % trials [1:10]
            for ch = 1:4 % neurons [(1:4),(1:4)]

                t_wt = eval(['tuningcurve_' labels{ICtype} '(ch,x == tloc(z))']);
                m_wt = eval(['tuningcurve_' labels{ICtype} '(ch,x == mloc(z))']);

                if isempty(t_wt), t_wt = 0; end
                if isempty(m_wt), m_wt = 0; end

                if t <= 10 %song 1
                    singleConfigSpks(t,ch,:) = t_wt.*fr_target{1} + m_wt.*fr_masker{t};
                else
                    singleConfigSpks(t,ch,:) = t_wt.*fr_target{2} + m_wt.*fr_masker{t-10};
                end

                if t_wt + m_wt >= 1
                    singleConfigSpks(t,ch,:) = singleConfigSpks(t,ch,:) / (t_wt + m_wt);
                end
            end
        end
        
        % format of spks is : [trial x channel x time]
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,4,padSize));
        end

        if options.nCells == 1, spks = cat(3,spks,singleConfigSpks(:,3,:));
        elseif options.nCells == 2, spks = cat(3,spks,singleConfigSpks(:,[1 3],:));
        elseif options.nCells == 3, spks = cat(3,spks,singleConfigSpks(:,[1 3 4],:));
        elseif options.nCells == 4, spks = cat(3,spks,singleConfigSpks);
        end

    end

    % format of spks should be : [time x channel x trial]
    spks = permute(spks,[3 2 1]);
    save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks');
end
end

%% run simulation

if isempty(options.locNum), options.time_end = size(spks,1)*dt; %ms;
else, options.time_end = padToTime; end
[snn_out,s] = columnNetwork(study_dir,varies,options,netcons);

%% post-process

% figure; plotsimPSTH('',snn_out);

% if numVaried <= 5
%     for n = 1:numVaried
%         plotUnitVoltage('C',snn_out,n);
%         saveas(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '.png']);
%         savefig(gcf,[simDataDir filesep 'C unit voltage, set ' num2str(n) '']);
%     end
% end

% load ICfiles struct just for the names of the configs
load('ICfiles.mat');
subz = 1:24;

% calculate performance
data = struct();
options.time_end = padToTime; %ms
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; %units of samples
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-options.time_end/dt+1);
configName = cellfun(@(x) strsplit(x,'_'),{ICfiles(subz).name}','UniformOutput',false);
configName = vertcat(configName{:}); configName = configName(:,1);
options.variedField = strrep(expVar,'-','_');
options.chansToPlot = [1 2];

annotTable = createSimNotes(snn_out,expName,options);

% save C spikes and varied params to struct
names = snn_out(1).varied; results = struct;
for i = 1:length(snn_out)
results(i).C_V_spikes = snn_out(i).C_V_spikes;

for t = 1:length(names)
results(i).(names{t}) = snn_out(i).(names{t});
end
end
results(1).model = snn_out(1).model;
save([simDataDir filesep 'C_results.mat'],'results');

tic;
if ~isempty(options.locNum)
    trialStart = 1; trialEnd = padToTime/dt;
    figName = [simDataDir filesep configName{options.locNum}(1:end-4)];
    [data.perf,data.fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
    plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options);
else
    for z = 1:24
        trialStart = PPtrialStartTimes(z);
        trialEnd = PPtrialEndTimes(z);
        figName = [simDataDir filesep configName{z}];
        [data(z).perf,data(z).fr] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options);
        plotRasterTree(snn_out,s,trialStart,trialEnd,figName,options);
    end
end
toc;
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
    dontPlot = {'ICE','ICI'};
    names(matches(names,dontPlot)) = [];
    
    for p = 1:length(names)
        plotPerfvsParams(names{p},data,varies,simDataDir)
    end
    close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results

figure; vizNetwork(s,0,'C','')
saveas(figure(1),fullfile('simData',expName,'netcons.png'));
saveas(figure(2),fullfile('simData',expName,'network.png'));
close all;


if isempty(options.locNum)

    [pc] = retrieveSPIKEPerf(results,numVaried);

    close all;
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
    close all;
    
    % cleaner grid
    for n = 1:numVaried
        clean_perf = pc(n,[5 10 15 20]);
        masked_perf = flipud(reshape(pc(n,[6 7 8 9 11:14 16:19 21:24]),[4 4]));
        plot_spatial_grid(['Simulation ' num2str(n)],clean_perf,masked_perf,50,90)
        savefig(gcf,[simDataDir filesep 'C, variation ' num2str(n) ', nice grid.fig'])
        % close;
    end

end
