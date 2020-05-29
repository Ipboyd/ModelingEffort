function [perf, fr, annotstr] = mouse_network(study_dir,time_end,varies,netCons,plot_rasters)
% [performance, tauMax] = mouse_network(study_dir,time_end,varies,plot_rasters,plot_title)
% study_dir: location of IC spike files + directory for log and data files
% time_end: length of simulation in ms
% varies: vary parameter, a structure. e.g.
%   varies(1).conxn = '(IC->IC)';
%   varies(1).param = 'trial';
%   varies(1).range = 1:20;
%   *first set of parameters should always be "trial" for IC->IC cxn
% plot_rasters: 1 or 0
% plot_title: so we know which files the figures correspond to 
%
% @Kenny F Chou, Boston Univ. 2019-06-20
% 2019-08-04 - added sharpening neurons
% 2019-08-14 - plotting now handles multiple varied parameters
% 2019-09-11 - removed redundant code. Return R performance in addition to
%              C performance
% 2020-02-05 - added netCons as parameter
% to do - fix plot_rasters option

%% Input check
if ~strcmp(varies(1).param,'trial')
    error('first set of varied params should be ''trial''')
end

%% solver params
solverType = 'euler';
dt = 1; %ms % the IC input is currently dt=1
viz_network = 0;

%% visualize IC spikes (Figure 1 which is the IR level (as seen from the
% inputguassian file)
%{
figure
for i = 1:4 %for each spatially directed neuron
   subplot(1,4,i)
   plotSpikeRasterFs(logical(squeeze(spks(:,i,:))), 'PlotType','vertline');
   xlim([0 time_end])
   line([0,time_end],[10.5,10.5],'color',[0.3 0.3 0.3])
end
%}

%% neuron populations
%tonic = bias = cells spontaneous firing

nCells = 4;
noise = 0.01; % low noise
s = struct();

s.populations(1).name = 'IC';
s.populations(1).equations = 'chouLIF';
s.populations(1).size = nCells;
s.populations(1).parameters = {'Itonic',0,'noise',0}; % 10-20 Hz spiking at rest

s.populations(end+1).name = 'I';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'Itonic',0, 'noise',0}; % 10-20 Hz spiking at rest

s.populations(end+1).name = 'S'; % S for sharpening
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'Itonic',0, 'noise',0}; % 10-20 Hz spiking at rest

s.populations(end+1).name='R';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = nCells;
s.populations(end).parameters = {'Itonic',0,'noise',0};

s.populations(end+1).name='C';
s.populations(end).equations = 'chouLIF';
s.populations(end).size = 1;
s.populations(end).parameters = {'noise',noise};

%% connection mechanisms
synDoubleExp={
  'gSYN = 1; ESYN = 0; tauD = 4; tauR = 1; delay = 0'
  'f(x) = (exp(-x/tauD) - exp(-x/tauR)).*(x>0)'
  'netcon=ones(N_pre,N_post)'
  'synDoubleExp(X,t) = gSYN .* ( f(t - tspike_pre - delay) * netcon ).*(X - ESYN)'
  '@isyn += synDoubleExp(V_post,t)'
  };

s.mechanisms(1).name='synDoubleExp';
s.mechanisms(1).equations=synDoubleExp;

%% connections
% build I->R netcon matrix
% netcons are [N_pre,N_post]

if isfield(netCons,'irNetcon'), irNetcon = netCons.irNetcon; else, irNetcon = zeros(nCells); end %no xchan inhibition by default
if isfield(netCons,'srNetcon'), srNetcon = netCons.srNetcon; else, srNetcon = diag(ones(1,nCells)); end %sharpening by default
if isfield(netCons,'rcNetcon'), rcNetcon = netCons.rcNetcon; else, rcNetcon = 'ones(N_pre,N_post)'; end

s.connections(1).direction='IC->IC';
s.connections(1).mechanism_list='IC';
s.connections(1).parameters={'g_postIC',0.16,'trial',5}; % 100 hz spiking

s.connections(end+1).direction='IC->I';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.21, 'tauR',0.4, 'tauD',2, 'netcon',diag(ones(1,nCells))}; 

s.connections(end+1).direction='IC->S';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.21, 'tauR',0.4, 'tauD',2, 'netcon',diag(ones(1,nCells))}; 

s.connections(end+1).direction='IC->R';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.21, 'tauR',0.4, 'tauD',2, 'netcon',diag(ones(1,nCells)),'delay',0}; 

s.connections(end+1).direction='I->R';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.2, 'tauR',0.4, 'tauD',10, 'netcon',irNetcon, 'ESYN',-80}; 

s.connections(end+1).direction='S->R';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.17, 'tauR',0.4, 'tauD',5, 'netcon',srNetcon, 'ESYN',-80, 'delay',3}; 

s.connections(end+1).direction='R->C';
s.connections(end).mechanism_list='synDoubleExp';
s.connections(end).parameters={'gSYN',.16, 'tauR',0.4, 'tauD',2, 'netcon',rcNetcon}; 

if viz_network, vizNetwork; end

%% vary params
vary = cell(length(varies),3);
for i = 1:length(varies)
    vary{i,1} = varies(i).conxn;
    vary{i,2} = varies(i).param;
    vary{i,3} = varies(i).range;
end

%% simulate
tic;
data = dsSimulate(s,'time_limits',[dt time_end], 'solver',solverType, 'dt',dt,...
  'downsample_factor',1, 'save_data_flag',1, 'save_results_flag',1,...
  'study_dir',study_dir, 'vary',vary, 'debug_flag',1, 'verbose_flag',0);
toc
%% insert spikes
V_spike = 50;
for iData = 1:length(data)
  for pop = {s.populations.name}
    pop = pop{1};
    data(iData).([pop '_V'])(data(iData).([pop '_V_spikes']) == 1) = V_spike; % insert spike
  end
end

%%
jump = length(find([data.IC_IC_trial]==1));
for vv = 1:jump % for each varied parameter
    subData = data(vv:jump:length(data));

    %% visualize spikes
    ICspks = zeros(20,4,time_end);
    Ispks = zeros(20,4,time_end);
    Rspks = zeros(20,4,time_end);
    Cspks = zeros(20,time_end);
    for i = 1:20
        for j = 1:4
            ICspks(i,j,:) = subData(i).IC_V_spikes(:,j);
            Ispks(i,j,:) = subData(i).I_V_spikes(:,j);
            Rspks(i,j,:) = subData(i).R_V_spikes(:,j);
        end
        Cspks(i,:) = subData(i).C_V_spikes;
    end

    % plot
    clf;
    for i = 1:4 %for each spatially directed neuron
        subplot(4,4,i+12)
        thisRaster = squeeze(ICspks(:,i,:));
        [perf.IC(i,vv),fr.IC(i,vv)] = calcPCandPlot(thisRaster,time_end,1,plot_rasters);        
        if i==1, ylabel('IC'); end

        subplot(4,4,i+8)
        thisRaster = squeeze(Ispks(:,i,:));
        calcPCandPlot(thisRaster,time_end,0,plot_rasters);        
        if i==1, ylabel('I'); end
        xticklabels([])

        subplot(4,4,i+4)
        thisRaster = squeeze(Rspks(:,i,:));
        [perf.R(i,vv),fr.R(i,vv)] = calcPCandPlot(thisRaster,time_end,1,plot_rasters);
        if i==1, ylabel('R'); end
        xticklabels([])
    end
    subplot(4,4,2)
    [perf.C(vv),fr.C(vv)] = calcPCandPlot(Cspks,time_end,1,plot_rasters);  
    ylabel('C spikes')
    xticklabels([])

    % figure annotations
    FR_C = mean(sum(Cspks,2))/time_end*1000;
    paramstr = {data(1).varied{2:end}};
    annotstr{vv,1} = ['FR_C = ' num2str(FR_C)];
    annotstr{vv,2} = ['Disc = ' num2str(mean(max(perf.C(vv))))];
    for aa = 1:length(varies)-1
        annotstr{vv,aa+2} = sprintf('%s = %.3f',paramstr{aa},eval(['data(' num2str(vv) ').' paramstr{aa}]));
    end
    annotation('textbox',[.55 .85 .2 .1],...
               'string',annotstr(vv,:),...
               'FitBoxToText','on',...
               'LineStyle','none')

    parts = strsplit(study_dir, filesep);
    DirPart = fullfile(parts{1:end-1});
    saveas(gca,[filesep DirPart filesep parts{end} '_v2_' num2str(vv) '.tiff'])
end
end

function [pc,fr] = calcPCandPlot(raster,time_end,calcPC,plot_rasters,h)
    PCstr = '';
    if calcPC
        % spks to spiketimes in a cell array of 10x2
        tau = linspace(1,30,100);
        spkTime = cell(20,1);
        for ii = 1:20, spkTime{ii} = find(raster(ii,:)); end
        spkTime = reshape(spkTime,10,2);
        % calculate distance matrix & performance
        distMat = calcvr(spkTime, tau);
        [performance, ~] = calcpc(distMat, 10, 2, 1,[], 'new');
        pc = mean(max(performance));
        PCstr = ['PC = ' num2str(pc)];
        tauMax = mean(tau(max(performance)==performance));
    end
    
    %plot
    if plot_rasters
    plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline');
    fr = mean(sum(raster,2))/time_end*1000;
    title({PCstr,['FR = ' num2str(fr)]});
    xlim([0 time_end])
    line([0,time_end],[10.5,10.5],'color',[0.3 0.3 0.3])
    end
end