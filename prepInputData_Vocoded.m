
locs = [90 45 0 -90];

% z : 1-4 , masker only
% z : 5,10,15,20 , target only
% z : 6-9 and etc. , mixed trials

% load('FR_traces_search.mat');
tmax = numel(fr_target_on);

x = -108:108;

% excitatory tuning curves
tuningcurve(1,:) = 0.8*gaussmf(x,[24 0]) + 0.2;

% only create spks file if not done yet

if ~exist(fullfile(study_dir,'solve','IC_spks_on.mat'),'file') || options.regenSpks

    labels = {'on','off'};

    for ICtype = [1 2]
        % divide all times by dt to upsample the time axis
        spks = [];

        % load fr traces, scale by weights
        % dt = 0.1 ms

        singleConfigSpks = zeros(nTrials,1,tmax);

        for t = 1:nTrials % trials [1:10]
            for ch = 1 % neurons [(1:4),(1:4)]

                t_wt = tuningcurve(ch,x == 0);

                if isempty(t_wt), t_wt = 0; end

                fr_trace = eval(['fr_target_' labels{ICtype}]); % +1 because we're skipping the noise-only response
                singleConfigSpks(t,ch,1:length(fr_trace)) = t_wt .* fr_trace;
            end
        end

        % format of spks is : [trial x channel x time]
        % pad each trial to have duration of timePerTrial
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(nTrials,1,padSize));
        end
        spks = cat(3,spks,singleConfigSpks(:,1,:));

        % apply strfGain
        spks = spks * strfGain;

        % format of spks should be : [time x channel x trial]
        spks = permute(spks,[3 2 1]);
        save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks');
    end
end