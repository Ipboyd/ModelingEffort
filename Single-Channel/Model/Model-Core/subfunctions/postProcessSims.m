profile on;
% trialStartTimes and trialEndTimes need to be the same cumsum as the trial
% lengths

% trialStartTimes = zeros(1,length(subz)); %ms
% for i = 1:length(subz) 
%     trialStartTimes(i) = padToTime; %3500 ms
% end
subz = 15; %Temp
trialStartTimes = padToTime * ones(1, length(subz)); %ms


% start slicing data by trial lengths
options.trial_length = padToTime; % [ms]
PPtrialStartTimes = [1 cumsum(trialStartTimes)/dt+1]; % in [samples]
PPtrialEndTimes = PPtrialStartTimes(2:end)-(padToTime/dt-padToTime/dt+1);

% ICfiles contains names for spatial grid configs
load('ICfiles.mat','ICfiles');
configName = cellfun(@(x) x(1:end-4),{ICfiles.name}','UniformOutput',false);

% find varied parameter, excluding trials (1st entry in varies)
% if varied_param is empty, settle on 2nd entry in varies
varied_param = find( (cellfun(@length,{varies(2:end).range}) > 1 & ~cellfun(@iscolumn,{varies(2:end).range}))) + 1;
if isempty(varied_param), varied_param = 2; end
expVar = [varies(varied_param(1)).conxn '_' varies(varied_param(1)).param];
options.variedField = replace(expVar,'->','_');

%annotTable = createSimNotes(snn_out,simDataDir,options);

% save output spikes and varied params to struct
names = snn_out(1).varied; results = struct;

if any(strcmp({s.populations.name},'C'))
    output_spks = 'C_V_spikes';
    output_pop = 'C';
else
    output_spks = 'R2On_V_spikes';
    output_pop = 'R2On';
end

for i = 1:length(snn_out)
    results(i).(output_spks) = snn_out(i).(output_spks);
    for t = 1:length(names)
        results(i).(names{t}) = snn_out(i).(names{t});
    end
end
%results(1).model = snn_out(1).model; save([simDataDir filesep output_pop '_results.mat'],'results');

%% calculate discriminability and firing rate at each configuration
data = struct();

% calculate number of parameter sets (excluding repeats for optogenetic
% trials)
nVaried = length(snn_out)/(20*nSims);


%Should probably be parforing out here since we have more than 1 thing that
%we are diong out here.

%tic;

for i = 1:length(subz)
    data(subz(i)).perf = [];
    data(subz(i)).fr = [];
    data(subz(i)).spks = [];
end


%Note: The following function call was changed slightly. When interfacing
%with Julia to do MI calculation it wastes a lot of time. No we just
%calculate the performace fr on the last trial in one batch. This just
%saves time communicating with Julia
last_flag = false;

distMats = [];

% % Preallocate the data structure (if it isn't already)
% data(subz) = struct('perf', [], 'fr', [], 'spks', [], 'perfmed', [], 'config', []);

% Temporary variables for storing results
% perf_temp = cell(1, numel(subz));
% fr_temp = cell(1, numel(subz));
% spks_temp = cell(1, numel(subz));
% perfmed_temp = cell(1, numel(subz));
% config_temp = cell(1, numel(subz));

%Going to try to parfor this for speed on the MI calculation
for i = 1:length(subz)
    trialStart = PPtrialStartTimes(i); trialEnd = PPtrialEndTimes(i);
    %figName = [simDataDir filesep configName{subz(i)}];
    
    %Setup should be something like the following:
    
    %Test if we are on the last in subz.

    %For all but last subz
    
    % output_perf = [];
    % 
    % if(i == length(subz))
    %     last_flag = true; 
    % 
    %     %For the final subz
    %     [output_perf , data(subz(i)).fr , data(subz(i)).spks, data(subz(i)).perfmed,distMat] = ...
    %         postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options, plot_all,last_flag,distMats);
    %     data(subz(i)).config = configName{subz(i)};
    % 
    %     distMats = cat(3,distMats,distMat);
    % 
    % else
    % 
    %     [~, data(subz(i)).fr,data(subz(i)).spks,~,distMat] = postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options, plot_all,last_flag,distMats);
    %     data(subz(i)).config = configName{subz(i)};
    % 
    %     distMats = cat(3,distMats,distMat);
    % end
    
    figName = 'temp'; 

    [data(subz(i)).perf , data(subz(i)).fr , data(subz(i)).spks, data(subz(i)).perfmed] = ...
            postProcessData_new(snn_out,s,trialStart,trialEnd,figName,options, plot_all);
    data(subz(i)).config = configName{subz(i)};

    % [perf_temp{i}, fr_temp{i}, spks_temp{i}, perfmed_temp{i}] = ...
    %     postProcessData_new(snn_out, s, trialStart, trialEnd, figName, options, plot_all);
    % config_temp{i} = configName{subz(i)};

    

    % tree-plotting functions: makes figures for all units for each config

    %plotRasterTree(data(subz(i)),figName,options); %close;
    %plotPSTHTree(data(subz(i)),figName,options); %close; 

    % make PSTH from spks

    t_bin = 20; % in [ms]
    psth_vec = (300:t_bin:(300 + 3000))/dt;
    for vv = 1:nVaried
        for tid = 1:2
            raster = data(subz(i)).spks.(output_pop)(vv).channel1((1:10) + 10*(tid-1),:);
            [~,spk_inds] = find(raster);
            data(subz(i)).output_PSTH(tid,:,vv) = histcounts(spk_inds,psth_vec);
        end
    end

    if nVaried == 1
        data(subz(i)).output_PSTH = squeeze(data(subz(i)).output_PSTH);
    end
end
%toc;


%Fix things since we split up finding the 


% Initialize arrays before the parfor loop
% data_perf = cell(1, length(subz));
% data_fr = cell(1, length(subz));
% data_spks = cell(1, length(subz));
% data_output_PSTH = cell(1, length(subz));
% data_config = cell(1, length(subz));


%Go through and remove all the things that we do not need to pass into the
%next part


%Temporary adding this so that we can work with the single channel model.
%(plot all condition^^)
if plot_all == 0

    all_fields = fieldnames(snn_out);
    for fields_input = 1:length(fieldnames(snn_out))
        if(~strcmp(all_fields{fields_input},'ROn_V_spikes') && ~strcmp(all_fields{fields_input},'C_V_spikes')  && ~strcmp(all_fields{fields_input},'On_V_spikes')  && ~strcmp(all_fields{fields_input},'X_V_spikes')...
          && ~strcmp(all_fields{fields_input},options.variedField) && ~strcmp(all_fields{fields_input},'model')...
          && ~strcmp(all_fields{fields_input},'On_On_trial'))
            
            snn_out = rmfield(snn_out, all_fields{fields_input});
    
        end
    end

end


% for i = 1:length(subz)
%     trialStart = PPtrialStartTimes(i); 
%     trialEnd = PPtrialEndTimes(i);
% 
%     figName = [simDataDir filesep configName{subz(i)}];
%     [data_perf{i}, data_fr{i}, data_spks{i}] = ...
%         postProcessData_new(snn_out, s, trialStart, trialEnd, figName, options,plot_all);
%     data_config{i} = configName{subz(i)};
% 
%     % tree-plotting functions: makes figures for all units for each config
%     % plotRasterTree(data(subz(i)),figName,options); %close;
%     plotPSTHTree(data(subz(i)),figName,options); %close; 
% 
%     % make PSTH from spks
%     t_bin = 20; % in [ms]
%     psth_vec = (300:t_bin:(300 + 3000))/dt;
%     output_PSTH = zeros(2, length(psth_vec) - 1, nVaried);
%     for vv = 1:nVaried
%         for tid = 1:2
%             raster = data_spks{i}.(output_pop)(vv).channel1((1:10) + 10*(tid-1), :);
%             [~, spk_inds] = find(raster);
%             output_PSTH(tid, :, vv) = histcounts(spk_inds, psth_vec);
%         end
%     end
% 
%     if nVaried == 1
%         output_PSTH = squeeze(output_PSTH);
%     end
%     data_output_PSTH{i} = output_PSTH;
% end
% 
% % Store results back into the data structure
% for i = 1:length(subz)
%     data(subz(i)).perf = data_perf{i};
%     data(subz(i)).fr = data_fr{i};
%     data(subz(i)).spks = data_spks{i};
%     data(subz(i)).output_PSTH = data_output_PSTH{i};
%     data(subz(i)).config = data_config{i};
% end

% calculate control and laser performance for optogenetic trials
%if nSims >= 2
%    calcMeanOptoPerf(results,nVaried,simDataDir);
%end
%%


%For GA we are focusing on SPIKE performance for now. This is calculated
%already in postprocesses_new. This block does not inform our current
%output.
% pc = struct;
% for i = 1:length(subz)
%     [perf,FR] = calcPerfsandFR( data(subz(i)).spks , nVaried ,options.dt , output_pop);
%     pc(subz(i)).fr = FR;
%     pc(subz(i)).perf = perf;
%     pc(subz(i)).config = configName{subz(i)};
% end
% save([simDataDir filesep 'perf_fr_' output_pop '.mat'],'pc');

% make surface plot for 2D parameter searches
% if nVaried >= 10
%     for i = 1:length(subz)
%         plotPerfvsParams(output_pop,data(subz(i)),varies,simDataDir,data(subz(i)).config)
%         %close all;
%     end
% end

dist_measures = struct;



for i = 1:length(subz)
   % calculate trial similarity and RMS difference at output for each
    % config
    clearvars outputSpks_cell
    for vv = 1:nVaried
        for ns = 1:nSims
            % if the data(indexing) doesn't work replace it with the snn_out
            % version: {snn_out((vv + (nS-1)*nVaried) : nVaried*nSims : end).C_V_spikes(trialStart:trialEnd)}
            outputSpks = data(subz(i)).spks.(output_pop)(vv).channel1;
            for n = 1:20
                outputSpks_cell{n} = find(outputSpks(n,:))' * dt/1000 - 0.3;
            end
            outputSpks_cell = reshape(outputSpks_cell,10,2);
            [dist_measures(subz(i)).TS(vv,ns),dist_measures(subz(i)).RMS(vv,ns)] = calcTrialSim(outputSpks_cell);
        end
    end
    dist_measures(subz(i)).config = configName{subz(i)};
end

%save([simDataDir filesep 'TS_RMS_' output_pop '.mat'],'dist_measures')

%% Make full grid if we ran all spatial grid configurations

% if the locNum field is empty, the simulation runs all configurations
% (including masker-only) by default

% the second condition is if we run all configurations that have a target
% playing (will need to implement cases where locNum is an array instead of
% a single value)
if numel(subz) > 1
    simOptions.subz = subz;
    simOptions.locationLabels = strtrim(cellstr(num2str(locs'))');
    simOptions.chanLabels = chanLabels;
    if plot_all == 0
        subPops = {'C'};
    else
        subPops = {'C','On','ROn'};
    end
    %Taking out ROn to help improve speed 7/19 IB
    %subPops = {'C'};
    targetIdx = 5:5:20;
    mixedIdx = setdiff(1:24,[1:4 targetIdx]);

    [approximate_grid_cur,fr_grid_cur] = plotPerformanceGrids_v3(data,s,annotTable,subPops,targetIdx,mixedIdx,simOptions,expName, plot_all);
    approximate_grid = approximate_grid + approximate_grid_cur;
    fr_grid = fr_grid + fr_grid_cur;
end

%disp('here')

%profile off;
% profile viewer

%% peakDrv for spatial grid stimuli

% ignore this for making full spatial grids

% getPeakDrv_SpatialStim;
profile off;
%profile viewer;