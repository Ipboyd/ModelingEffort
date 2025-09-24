
clear
close all

%load('/Users/lbowman/Desktop/Research/sound_files.mat','sampleRate','target1','target2');

cd(userpath);
cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')


load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

t1 = abs(target1);
st1 = smoothdata(t1,'gaussian',25000);

%% Find onset/offset regions in target 1

% Find low peaks

is_minima = islocalmin(st1);

peak_values = st1(is_minima);
peak_locations = find(is_minima);
min_values = peak_values .* (peak_values < mean(st1));

min_loc = [];
index = 1;

for i = 1:length(min_values)
    if min_values(i) > 0
        min_loc(index) = peak_locations(i);
        index = index +1;
    end
end
min_loc = min_loc';
min_values = min_values(min_values ~= 0);


figure;
scatter(min_loc,min_values)
hold on
plot(st1)
hold off

% Define onset cutoff

mean_cutoff = mean(min_values);

on_st1 = st1 > mean_cutoff;
fon_st1 = on_st1 .* round(max(st1),2);

% figure;
% b = bar(fon_st1);
% hold on
% plot(st1,'LineWidth',1,'Color','b')
% hold off
% b.FaceAlpha = 0.2;
% b.FaceColor = 'b';
% title('Onset Regions of Target Audio Above Cutoff Magnitude')
% legend('Onset Region','Target 1')

angles = {'n90','p0','p45','p90'};
empty_struct = struct('animal_names',{all_data.subject},'n90','','p0','','p45','','p90','');

for i = 1:length(empty_struct)
    empty_struct(i).n90 = struct('is_Onset','','is_Offset','','is_Both','','is_Neither','','Onset_Data','','Offset_Data','');
    empty_struct(i).p0 = struct('is_Onset','','is_Offset','','is_Both','','is_Neither','','Onset_Data','','Offset_Data','');
    empty_struct(i).p45 = struct('is_Onset','','is_Offset','','is_Both','','is_Neither','','Onset_Data','','Offset_Data','');
    empty_struct(i).p90 = struct('is_Onset','','is_Offset','','is_Both','','is_Neither','','Onset_Data','','Offset_Data','');
end

%% Plot PSTH and store in struct

total_spike_data = [];

for n = 7 % Number of cells

    for ang = 1:3 % Number of angles

        figure;

        for k = 1:10

            SpikeTimes = all_data(n).ctrl_tar1_timestamps{k,ang};    %(10 x 4) As it stands (look at animal at index 124 and the timestamps associated with the first location and iterate over all 10 trials)
            b = transpose(repmat(SpikeTimes,1,3));                 %Reshape SpikeTimes
            y_lines = nan(1,length(SpikeTimes));                   %Create a vertical line at each spike time
            y_lines(1,:) = k-1;
            y_lines(2,:) = k;
            y_lines(3,:) = nan;
            h2 = plot(b,y_lines,'Color',[0 0 0 0.8],'LineWidth',0.4); hold on   %Plot
            total_spike_data = [total_spike_data; SpikeTimes];     %Save data for PSTH plotting

        end

        counts = histcounts(total_spike_data,'BinWidth',0.04);   %Bin width of 40ms
        domain = linspace(-1,4,length(counts));

        start = 0;
        stop = 0;

        for i = 1:length(domain)
            if domain(i) > 0
                start = i;
                break
            end
        end

        for i = 1:length(domain)
            if domain(i) >= 3
                stop = i;
                break
            end
        end

        neg_domain = domain(1:start-1);  % Store for z-score calc later
        neg_counts = counts(1:start-1);

        domain = domain(start:stop-1);
        counts = counts(start:stop-1);

        slen = length(fon_st1);
        sfac = 3/slen;
        sx = (sfac:sfac:3)';

        fac_on_st1 = fon_st1 .* (max(counts)+1)/max(fon_st1);
        fst1 = st1 .* max(counts)/max(st1);

        % Plot PSTH & onset regions

        figure;
        hold on
        plot(sx,fst1,'LineWidth',0.5,'Color',[0 0 1 0.3])
        a = area(sx,fac_on_st1);
        plot(domain,counts,'Color','r','LineWidth',1)
        hold off
        a.FaceAlpha = 0.1;
        a.FaceColor = 'b';
        a.EdgeColor = "none";
        title('PSTH with Onset Regions')
        legend('Target 1','Onset Region','PSTH')

        %% New way to find maxes

        onoff = double(on_st1);
        onsets = [];
        offsets = [];

        for k = 1:length(onoff)-1
            if (onoff(k) == 0 && onoff(k+1) == 1)
                onsets = [onsets,sx(k)];
            elseif (onoff(k) == 1 && onoff(k+1) == 0)
                offsets = [offsets,sx(k)];
            end
        end

        range = [];
        on_maxes = [];
        onset_points = zeros(length(onsets),4);

        for i = 1:length(onsets)

            range = [];

            idx = find(domain >= onsets(i) & domain <= offsets(i));
            onidx1 = min(idx);
            onidx2 = max(idx);

            onset_points(i,3) = onidx1;
            onset_points(i,4) = onidx2;

            for j = onidx1:onidx2
                range = [range, counts(j)];
            end

            maxp = max(range);
            on_maxes = [on_maxes, maxp];

            onset_points(i,1) = onsets(i);
            onset_points(i,2) = maxp;

        end

        range2 = [];
        off_maxes = [];
        offset_points = zeros(length(offsets),4);

        for i = 1:length(offsets)-1

            range2 = [];

            idx = find(domain >= offsets(i) & domain <= onsets(i+1));
            offidx1 = min(idx);
            offidx2 = max(idx);

            for j = offidx1:offidx2
                range2 = [range2, counts(j)];
            end

            maxp = max(range2);
            off_maxes = [off_maxes, maxp];

            offset_points(i,1) = offsets(i);
            offset_points(i,2) = maxp;
            offset_points(i,3) = offidx1;
            offset_points(i,4) = offidx2;

        end

        % final off_max

        for j = offidx2:length(domain)
            range2 = [range2, counts(j)];
        end

        maxp = max(range2);
        off_maxes = [off_maxes, maxp];

        offset_points(length(offsets),1) = offsets(length(offsets));
        offset_points(length(offsets),2) = maxp;
        offset_points(length(offsets),3) = offidx2;
        offset_points(length(offsets),4) = length(domain);

        %% Checking on maxes

        % || 1) Must be > 0.025 spikes/bins/trials ||

        % Crop spike data to only 0-3 sec

        crop_start_spikes = total_spike_data > 0;
        crop_start_spikes = crop_start_spikes .* total_spike_data;

        crop_end_spikes = crop_start_spikes <= 3;
        crop_spikes = crop_end_spikes .* crop_start_spikes;
        crop_spikes = crop_spikes > 0;
        num_spikes = sum(crop_spikes);

        % Find qualifying onset maxes

        num_bin = length(counts);
        num_trials = 10;

        min_max_denom = num_spikes/num_bin/num_trials;
        min_max_height = 0.025 * min_max_denom;

        qual_height_on_maxes = (on_maxes > min_max_height)';

        % || 2) Must be > 3x mean preceeding 50 ms ||

        % Go back 50ms from max to check mean

        for i = 1:length(onset_points)

            prev_edge = onset_points(i,1) - 0.05;

            if (onset_points(i,1) - 0.05) < 0
                prev_edge = 0;
            end

            % find closest domain/counts index to prev edge
            diffs = abs(domain - prev_edge);
            [min_diff, prev_index] = min(diffs);

            curr_index = find(counts == onset_points(i,2));

            % find only most relevant curr_index value
            if length(curr_index) > 1
                mci = find(curr_index > prev_index,1);
                curr_index = curr_index(mci);
            end
            prev_mean = mean(counts(prev_index:curr_index));
            qual_mean_on_maxes = onset_points(:,2) >= 3*prev_mean;

        end

        % || 3) Must be >= mean + 3sd preceeding signal noise before 0s ||

        % Mean + 3sd

        neg_spike_mean = mean(neg_counts);  % stored PSTH counts before 0
        sd_neg_counts = std(neg_counts);
        z_score_fac = neg_spike_mean + 3 * sd_neg_counts;

        % Qualifying on maxes

        qual_zscore_on_maxes = (on_maxes > z_score_fac)';

        % Array of all onset points and if they qualify

        qual_on_points = [];

        qual_on_points = [onset_points(:,1),onset_points(:,2),...
            double(qual_height_on_maxes),double(qual_mean_on_maxes), ...
            double(qual_zscore_on_maxes)];
        
        % Add to struct
        % Find angle for struct

        is_Onset = 0;

        for h = 1:length(qual_on_points)
            if sum(qual_on_points(h,3:5)) == 3
                is_Onset = 1;
                break
            end
        end

        if ang == 1
            empty_struct(n).n90.Onset_Data = qual_on_points;
        elseif ang == 2
            empty_struct(n).p0.Onset_Data = qual_on_points;
        elseif ang == 3
            empty_struct(n).p45.Onset_Data = qual_on_points;
        else
            empty_struct(n).p90.Onset_Data = qual_on_points;
        end


        %% Checking off maxes

        % Begin checking on-maxes based on 3 qualifications
        % || 1) Must be > 0.025 spikes/bins/trials ||

        qual_height_off_maxes = (off_maxes > min_max_height)';

        % || 2) Must be > 3x mean preceeding 50 ms ||

        % Go back 50ms from max to check mean

        for i = 1:length(offset_points)

            prev_edge = offset_points(i,1) - 0.05;

            if (offset_points(i,1) - 0.05) < 0
                prev_edge = 0;
            end

            % find closest domain/counts index to prev edge
            diffs = abs(domain - prev_edge);
            [min_diff, prev_index] = min(diffs);


            curr_index = find(counts == offset_points(i,2));

            % find only most relevant curr_index value
            if length(curr_index) > 1
                mci = find(curr_index > prev_index,1);
                curr_index = curr_index(mci);
            end
            prev_mean = mean(counts(prev_index:curr_index));
            qual_mean_off_maxes = offset_points(:,2) >= 3*prev_mean;

        end

        % || 3) Must be >= mean + 3sd preceeding signal noise before 0s ||

        qual_zscore_off_maxes = (off_maxes > z_score_fac)';

        % Array of all offset points and if they qualify

        qual_off_points = [];

        qual_off_points = [offset_points(:,1),offset_points(:,2),...
            double(qual_height_off_maxes),double(qual_mean_off_maxes),...
            double(qual_zscore_off_maxes)];
        
        is_Offset = 0;
        is_Both = 0;
        is_Neither = 0;

        for h = 1:length(qual_off_points)
            if sum(qual_on_points(h,3:5)) == 3
                is_Offset = 1;
                break
            end
        end

        if (is_Onset == 1) & (is_Offset == 1)
            is_Both = 1;
        elseif (is_Onset == 0) & (is_Offset == 0)
            is_Neither = 1;
        end

        if ang == 1
            empty_struct(n).n90.Offset_Data = qual_off_points;
            empty_struct(n).n90.is_Offset = is_Offset;
            empty_struct(n).n90.is_Both = is_Both;
            empty_struct(n).n90.is_Neither = is_Neither;
            empty_struct(n).n90.is_Onset = is_Onset;
        elseif ang == 2
            empty_struct(n).p0.Offset_Data = qual_off_points;
            empty_struct(n).p0.is_Offset = is_Offset;
            empty_struct(n).p0.is_Both = is_Both;
            empty_struct(n).p0.is_Neither = is_Neither;
            empty_struct(n).p0.is_Onset = is_Onset;
        elseif ang == 3
            empty_struct(n).p45.Offset_Data = qual_off_points;
            empty_struct(n).p45.is_Offset = is_Offset;
            empty_struct(n).p45.is_Both = is_Both;
            empty_struct(n).p45.is_Neither = is_Neither;
            empty_struct(n).p45.is_Onset = is_Onset;
        else
            empty_struct(n).p90.Offset_Data = qual_off_points;
            empty_struct(n).p90.is_Offset = is_Offset;
            empty_struct(n).p90.is_Both = is_Both;
            empty_struct(n).p90.is_Neither = is_Neither;
            empty_struct(n).p90.is_Onset = is_Onset;
        end

    end
end

%close all


%%%% Figure out why onset/offset data is stacking!!

