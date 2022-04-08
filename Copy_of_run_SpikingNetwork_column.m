% inputs: user specified, raw IC output
% network structure: user specified

cd('U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid')
dynasimPath = '../DynaSim';

% % max firing rate at offset input is 85% of onset input max
% % ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\with_offset//alpha_0.010 N1_5 N2_7//s38_STRFgain0.17_20220224-165328';
% 
% % max offset firing rate = max onset firing rate
% ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\with_offset//alpha_0.010 N1_5 N2_7//s38_STRFgain0.17_20220301-104013';

% absolute refractory period of 0.5ms w/ relative refractory period of 1ms,
% steepeness recovery factor of 2.5

addpath('mechs'); addpath('fixed-stimuli');
addpath('genlib'); addpath('plotting'); addpath(genpath(dynasimPath));
addpath('cSPIKE'); InitializecSPIKE;

model = struct; model.type = 'On';
model.interaction = 0;

% debug_flag = 1; save_flag = 0;

ICdir = 'U:\eng_research_hrc_binauralhearinglab\noconjio\Grid simulation code (July 2021 - )\MouseSpatialGrid\ICSimStim\mouse\vary_recovery\alpha_0.010\STRFgain 0.10, 1.0ms tau_rel';

expName = ['testing relative refractory times/' num2str(t_rels(r)) 'ms'];

% setup directory for current simulation
datetime = datestr(now,'yyyymmdd-HHMMSS');
study_dir = fullfile(pwd,'run',datetime);
if exist(study_dir, 'dir'),rmdir(study_dir, 's'); end
mkdir(fullfile(study_dir, 'solve'));
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

options = struct;
options.nCells = 2;

dt = 0.1; %ms

% % % DO NOT CHANGE THIS % % %
varies(1).conxn = '(On->On,Off->Off)';
varies(1).param = 'trial';
varies(1).range = 1:20;
% % % DO NOT CHANGE THIS % % %

varies(end+1).conxn = '(On->On)';
varies(end).param = 'g_postIC';
varies(end).range = [ 0.265 ];

% % recurrent excitation of S cells
% varies(end+1).conxn = '(R1On->S1On,R2On->S2On,R1Off->S1Off,R2Off->S2Off)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.02 ];
% 
% % cross-column excitation of S cells
% varies(end+1).conxn = '(R1On->S1Off,R2On->S2Off,R1Off->S1On,R2Off->S2On)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.02 ];


% % feedforward inhibition from S cells
% varies(end+1).conxn = '(S1On->R1On,S2On->R2On,S1Off->R1Off,S2Off->R2Off)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.02 ];
% 
% % cross-column inhibition from S cells
% varies(end+1).conxn = '(S1On->R1Off,S2On->R2Off,S1Off->R1On,S2Off->R2On)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.02 ];


% % inputs to S cell from same layer
% varies(end+1).conxn = '(R1On->S1On,R2On->S2On,R1Off->S1Off,R2Off->S2Off,R1On->S1Off,R2On->S2Off,R1Off->S1On,R2Off->S2On)';
% varies(end).param = 'tauP';
% varies(end).range = [ 1 , 40 : 40 : 240 ];
% 
% % inhibition to R cells from same layer
% varies(end+1).conxn = '(S1On->R1On,S2On->R2On,S1Off->R1Off,S2Off->R2Off,S1On->R1Off,S2On->R2Off,S1Off->R1On,S2Off->R2On)';
% varies(end).param = 'tauP';
% varies(end).range = [ 1 , 40 : 40 : 240 ];

% % all synapses with depression
% varies(end+1).conxn = '(On->S1On,R1On->S2On,Off->S1Off,R1Off->S2Off,R1On->S1On,R2On->S2On,R1Off->S1Off,R2Off->S2Off,R1On->S1Off,R2On->S2Off,R1Off->S1On,R2Off->S2On,S1On->R1On,S2On->R2On,S1Off->R1Off,S2Off->R2Off,S1On->R1Off,S2On->R2Off,S1Off->R1On,S2Off->R2On)';
% varies(end).param = 'fP';
% varies(end).range = [ 0 : 0.1 : 1 ];

% % inputs to X cells
% varies(end+1).conxn = '(R1On->X1On,R2On->X2On,R1Off->X1Off,R2Off->X2Off)';
% varies(end).param = '(gSYN)';
% varies(end).range = [ 0 0.0035 ];

% % cross-channel inhibition to R cells
% varies(end+1).conxn = '(X1On->R1On,X2On->R2On,X1Off->R1Off,X2Off->R2Off)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.0035 ];


% % inhibitory convergence to output cell
% varies(end+1).conxn = '(R2On->C,R2Off->C)';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0.008 ];

% % inhibitory convergence to output cell
% varies(end+1).conxn = 'S2On->C';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0 : 0.005 : 0.02 ];
% 
% varies(end+1).conxn = 'S2Off->C';
% varies(end).param = 'gSYN';
% varies(end).range = [ 0 : 0.005 : 0.02 ];

% control and opto conditions
varies(end+1).conxn = '(S1On,S1Off)';
varies(end).param = 'Itonic';
varies(end).range = [0 -0.1];

varies(end+1).conxn = '(R2On->R2On,R2Off->R2Off)';
varies(end).param = '(FR,sigma)';
varies(end).range = [ 8 10 ; 6 6 ];

% display parameters
network_params = [{varies.conxn}' {varies.param}' {varies.range}'];

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
end

expVar = [varies(varied_param).conxn '-' varies(varied_param).param];
expVar = strrep(expVar,'->','_');
numVaried = length(varies(varied_param).range);

%% prep input data
% concatenate spike-time matrices, save to study dir
trialStartTimes = zeros(1,length(subz)); %ms
padToTime = 3500; %ms
label = {'On','Off'};
for ICtype = [0 1] % only E no I
    % divide all times by dt to upsample the time axis
    spks = [];
    for z = 1:length(subz)
        % disp(ICfiles(subz(z)+0).name); %read in E spikes only
        load([ICdir filesep ICfiles(subz(z)).name],'t_spiketimes_on','t_spiketimes_off');
        
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
        singleConfigSpks(:,2,:) = 0; % zero out the U channel
        
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
        end
        
    end
    save(fullfile(study_dir,'solve',sprintf('IC_spks_%s.mat',label{ICtype+1})),'spks');
end

%% run simulation
options.ICdir = ICdir;
options.STRFgain = extractBetween(ICdir,'gain','_202');
options.plotRasters = 0;

% all locations
% options.time_end = size(spks,3)*dt; %ms;
% options.locNum = [];

options.time_end = padToTime;

if options.nCells == 1, options.locNum = 15; % 15 = clean target at 0deg
else, options.locNum = 16; % 16 = target at 0deg, masker at 90deg
end
% options.locNum = 18; % colocated at 0deg
% options.locNum = 8; % target at 90deg, masker at 0deg
options.locNum = 15;

options.parfor_flag = 0;

[snn_out,s] = columnNetwork(study_dir,varies,options,netcons,model);

%% post-process

% figure; 
% plotsimPSTH('On',snn_out); plotsimPSTH('R1On',snn_out); plotsimPSTH('R2On',snn_out);
% legend('On','R1On','R2On');

% if numVaried <= 5
%     for n = 1:numVaried
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

annotTable = createSimNotes(snn_out,expName,model);

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
close all;

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

[pc,fr] = plotParamvsPerf_1D(results);

% temp = struct2cell(pc);
% 
% ctrl = cellfun(@(x) x(1),temp);
% laser = cellfun(@(x) x(2),temp);
% 
% figure;
% bar((1:4)-.125,ctrl,0.25); hold on;
% bar((1:4)+.125,laser,0.25); ylim([50 100]); set(gca,'xticklabels',{'SPIKE','ISI','RI-SPIKE','Spike count'},'xtick',[1:4]);
% ytickformat('percentage');
% ylabel('Performance'); legend('Control','Laser');
% savefig(gcf,[simDataDir filesep 'Control vs laser performance.fig']);
% close all;

% grid search plots
if numVaried > 10
    names = {s.populations.name};
    dontPlot = {'On','Off'};
    names(matches(names,dontPlot)) = [];
    
    for p = 1:length(names)
        plotPerfvsParams(names{p},data,varies,simDataDir)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results

% close all
% 
% figure; vizNetwork(s,0,'C','On')
% saveas(figure(1),fullfile('simData',expName,'netcons.png'));
% saveas(figure(2),fullfile('simData',expName,'network.png'));
% close all;
% 
% targetIdx = find(cellfun(@(x) contains(x,'m0'),{ICfiles.name}));
% mixedIdx = find(cellfun(@(x) ~contains(x,'m0') && ~contains(x,'s0'),{ICfiles.name}));

if isempty(options.locNum)
    
    simOptions = struct;
    
    simOptions.subz = subz;
    simOptions.varies = varies;
    simOptions.varied_param = varied_param;
    simOptions.locationLabels = {'90','45','0','-90'};
    simOptions.expVar = expVar;
    simOptions.chanLabels = simOptions.locationLabels;
    
    options.subPops = {'R','C'};
    plotPerformanceGrids_v3(data,s,annotTable,options.subPops,targetIdx,mixedIdx,simOptions,expName);
    
end
