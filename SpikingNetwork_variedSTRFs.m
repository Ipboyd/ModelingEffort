cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid-simulation-code\MouseSpatialGrid')
dynasimPath = '../DynaSim';

addpath('mechs'); addpath('fixed-stimuli'); addpath(genpath('ICSimStim'));
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;
addpath('plotting');

expName = '07-08-2022 seay parameters';
load('default_STRF_with_offset.mat');

gainFac = 1;
fr_target{1} = fr_target{1}*gainFac;
fr_target{2} = fr_target{2}*gainFac;
for m = 1:10
fr_masker{m} = fr_masker{m}*gainFac;
end

options = struct;
options.nCells = 1;
options.opto = 0; if options.opto, nSims = 5; else, nSims = 1; end

options.mex_flag = 0;
options.parfor_flag = 0;
options.plotRasters = 0;

% 1 for spatial attention, 0 for passive condition
options.SpatialAttention = 0;
% options.Imask = [0];

% locNum should be empty for full grids
options.locNum = 15;

study_dir = fullfile(pwd,'run','single-channel-V4');

if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));

simDataDir = [pwd filesep 'simData' filesep expName];
if ~exist(simDataDir,'dir'), mkdir(simDataDir); end

%% define network parameters
clear varies

dt = 0.1; %ms

trialInds = repmat(1:20,nSims,1);

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = '(IC->IC)';
varies(1).param = 'trial';
varies(1).range =  trialInds(:)';
% % % DO NOT CHANGE THIS % % %

% excitation
varies(end+1).conxn = '(IC->R1,R1->R2,R2->C)';
varies(end).param = '(gSYN,fP,tauP)';
varies(end).range = [ 0.015 ; 0.01 ; 80 ];

% within-channel inhibition
varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = '(gSYN)';
varies(end).range = [ 0.022 ];

% within-channel inhibition
varies(end+1).conxn = '(S1->R1,S2->R2,S2->C)';
varies(end).param = '(tauP,fP)';
varies(end).range = [ 300 ; 0.2 ];

% feed-forward excitation of PV cells 
varies(end+1).conxn = '(IC->S1,R1->S2)';
varies(end).param = '(gSYN)';
varies(end).range = [ 0.02 ];

% feed-forward excitation of PV cells 
varies(end+1).conxn = '(IC->S1,R1->S2)';
varies(end).param = '(tauP,fP)';
varies(end).range = [ 200 ; 0.05 ];

% adaptation recovery times
varies(end+1).conxn = '(R1,R2,C)';
varies(end).param = '(tau_ad,t_ref_rel)';
varies(end).range = [ 100 ; 1.5]; % using v_reset to decrease burstiness

% adaptation recovery times
varies(end+1).conxn = '(R1,R2,C)';
varies(end).param = '(g_inc)';
varies(end).range = [ 0.0001 ]; % using v_reset to decrease burstiness

% % control and opto conditions
% varies(end+1).conxn = 'S2';
% varies(end).param = '(Itonic)';
% varies(end).range = [ 0 ]; 
% if options.opto
%     varies(end).range = [ 0 -0.05 ];
% end

varies(end+1).conxn = '(C->C)';
varies(end).param = 'FR';
varies(end).range = 5;
if options.opto
    varies(end).range = [ 5 8 ];
end

% find varied parameter, excluding trials
varied_param = find( (cellfun(@length,{varies.range}) > 1 & ~cellfun(@iscolumn,{varies.range})));
if numel(varies(1).range) == 20, varied_param(1) = []; % delete trial varies
    end

if isempty(varied_param) % if no varied params, settle on 2nd entry in varies
    varied_param = 2;
end

% netcons can't be put in the varies struct (Dynasim doesn't recognize it?);
% it needs to be put in a different struct

netcons = struct; % row = source, column = target
netcons.XRnetcon = zeros(options.nCells,options.nCells);

% for simplification: use 1st varied param for 2d searches
expVar = [varies(varied_param(1)).conxn '-' varies(varied_param(1)).param];
expVar = strrep(expVar,'->','_');

% calculate # parameter sets
numVaried = prod(cellfun(@(x) size(x,2),{varies(varied_param).range}));
if nSims == 5, numVaried = 2; end

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

% load('FR_traces_search.mat');
temp = cellfun(@numel,fr_target);
tmax = max(temp);

x = -108:108;

% excitatory tuning curves
tuningcurve_E(1,:) = 0.8*gaussmf(x,[24 0]) + 0.2;

% % inhibitory tuning curves
% tuningcurve_I(1,:) = 0.8*gaussmf(x,[24 0]) + 0.2;

% only create spks file if not done yet

% if ~exist([study_dir filesep 'solve' filesep 'IC_spks_E.mat'],'file')
labels = {'E','I'};
for ICtype = 1
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

        singleConfigSpks = zeros(20,1,tmax);
        
        for t = 1:20 % trials [1:10]
            for ch = 1 % neurons [(1:4),(1:4)]

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
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,1,padSize));
        end

        spks = cat(3,spks,singleConfigSpks(:,1,:));

    end

    % format of spks should be : [time x channel x trial]
    spks = permute(spks,[3 2 1]);
    save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks');
end
% end

%% run simulation

if isempty(options.locNum), options.time_end = size(spks,1)*dt; %ms;
else, options.time_end = padToTime; end
[snn_out,s] = columnNetwork(study_dir,varies,options,netcons);

%% post-process

% plotISIHistogram('C',snn_out);

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

    % performance
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

    % firing rate
    ctrl_mean = mean(fr(:,1));
    laser_mean = mean(fr(:,2));

    ctrl_se = std(fr(:,1))/sqrt(5);
    laser_se = std(fr(:,2))/sqrt(5);

    figure('unit','inches','position',[5 5 2 3]);
    bar(1-.2,ctrl_mean,0.4,'facecolor','none','linewidth',2); hold on;
    bar(1+.2,laser_mean,0.4,'facecolor','k','linewidth',2);
    xlim([0.4 1.6]);

    errorbar(1-.2,ctrl_mean,ctrl_se,'color','k','linestyle','none','linewidth',1); hold on;
    errorbar(1+.2,laser_mean,laser_se,'color','k','linestyle','none','linewidth',1);

    ylim([0 60]);
    set(gca,'xticklabels',{'Control','Laser'},'xtick',[],'fontsize',8);
    ylabel('Firing rate (Hz)'); legend('Control','Laser');
    saveas(gcf,[simDataDir filesep 'opto_FR_results.fig']);
end

[pc,fr] = plotParamvsPerf_1D(results,numVaried)
save([simDataDir filesep 'perf_fr_C.mat'],'pc','fr')

x = varies(varied_param).range;

figure;
plot(x,pc.SPIKE); hold on; plot(x,pc.ISI); plot(x,pc.RISPIKE)
legend('SPIKE','ISI','RI-SPIKE'); ylim([50 100]);
xlabel([varies(varied_param).conxn '_{' varies(varied_param).param '}']);
title(['Performance vs ' varies(varied_param).conxn '_{' varies(varied_param).param '}']);
ytickformat('percentage'); ylabel('Performance')


savefig(gcf,[simDataDir filesep 'perfs_vs_params.fig'])

