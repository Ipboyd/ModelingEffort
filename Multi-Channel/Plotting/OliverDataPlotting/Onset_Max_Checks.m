
clear
close all

%load('/Users/lbowman/Desktop/Research/sound_files.mat','sampleRate','target1','target2');

cd(userpath);
cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')


load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz

% figure;
% plot(target1)

t1 = abs(target1);

% figure;
% plot(t1)

st1 = smoothdata(t1,'gaussian',20000);

x = max(t1);
y = max(st1);

fac = x/y;
st1 = st1 .* fac;

figure;
plot(st1)

figure;
hold on
plot(t1)
plot(st1)
hold off
% 
% tx = 0:(length(st1)-1);
% tx = tx';

% figure;
% plot(tx,st1)

dt1 = diff(st1);
sdt1 = smoothdata(dt1,"movmedian",500);

figure;
plot(sdt1)
yline(0)

onoff = sdt1 > 0;
fonoff = onoff .* max(st1);

figure;
hold on
plot(fonoff,'r')
plot(t1,'b')
hold off

% counts = [];
% 
% for i = 1:length(sdt1)
% 
%     if dt1(i) > 0
%         if dt1(i+500) > 0
%             counts(i:(i+5000),1) = 1;
%             i = i+1000;
%         end
%     else 
%         counts(i,1) = 0;
%     end
% end
% 
% figure;
% plot(counts)
% ylim([-0.5 1.5])


onidx = find(onoff);
off = 1 - onoff;
offidx = find(off);


%-----------------
%This bit of code Brings in the PSTH

%Bring in the PSTH for ex subject 7

total_spike_data = [];

for k = 1:10
    
    SpikeTimes = all_data(7).ctrl_tar1_timestamps{k,1};  %(10 x 4) %As it stands (look at animal at index 124 and the timestamps associated with the first location and iterate over all 10 trials)
    b = transpose(repmat(SpikeTimes,1,3));                 %Reshape SpikeTimes
    y_lines = nan(1,length(SpikeTimes));                   %Create a vertical line at each spike time
    y_lines(1,:) = k-1;
    y_lines(2,:) = k;
    y_lines(3,:) = nan;
    h2 = plot(b,y_lines,'Color',[0 0 0 0.8],'LineWidth',0.4); hold on   %Plot
    total_spike_data = [total_spike_data; SpikeTimes];      %Save data for PSTH plotting

end



counts = histcounts(total_spike_data,'BinWidth',0.01); %Bin width of 10ms
domain = linspace(-1,4,length(counts));

%Trim these to be around the start and end of the signal (Figure out
%exactly later)

% counts = counts(26:100);
% domain = domain(26:100);

% Possible way to automate crop 

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

domain = domain(start:stop);
counts = counts(start:stop);

figure;
plot(domain,counts)

num_onoff = double(onoff);
s_num_onoff = smoothdata(num_onoff,'movmean',50);
onoff = s_num_onoff > 0;

slen = length(onoff);
sfac = 3/slen;
sx = sfac:sfac:3;

fonoff = onoff .* (max(counts)+1);

figure;
plot(domain,counts)
hold on
plot(sx,fonoff)
hold off
xlim([-0.1 3.1])
ylim([0 20])


%-----------------
%This bit of code finds the onsets and offsets bin edges

% for k = 1:length(onoff)-1
%     if((onoff(k) == 1 && onoff(k+1) == 0) || (onoff(k) == 0 && onoff(k+1) == 1))
% 
%         %Divide by the domain mismatch between PSTH and original sigmal in
%         %order to normalize
% 
%         %PSTH is of length 125 right now (de-hardcoed later)
%         binedges = [binedges,k/(length(onoff)/length(domain))];
% 
%     end
% 
% end

offsets = [];
onsets = [];
binedges = [];

for k = 1:length(onoff)-1
    if(onoff(k) == 1 && onoff(k+1) == 0)

        offsets = [offsets,k/(length(onoff)/length(domain))];

    elseif(onoff(k) == 0 && onoff(k+1) == 1)

        onsets = [onsets,k/(length(onoff)/length(domain))];

    end

end

%-----------------
%This bit of code finds the maximum within the bin edges

%Figure out if first onset precedes first offset

if(onsets(1) < offsets(1))
    flag = 0;
else
    flag = 1;
end


on_maxes = [];
off_maxes = [];

for k = 1:length(onsets)
    
    rangeofvalsonset = [];

    %Edge case 1
    s_onset = floor(onsets(k));
    e_onset = ceil(onsets(k));

    where_on_the_slope = onsets(k)-s_onset;

    excat_val = (counts(e_onset)-counts(s_onset))*where_on_the_slope  + counts(s_onset);

    rangeofvalsonset = [rangeofvalsonset,excat_val];


    %Edge case 2
    s_offset = floor(offsets(k+flag));
    e_offset = ceil(offsets(k+flag));


    where_on_the_slope2 = offsets(k)-s_offset;

    excat_val2 = (counts(e_offset)-counts(s_offset))*where_on_the_slope2 + counts(s_offset);

    rangeofvalsonset = [rangeofvalsonset,excat_val2];

    
    %Everything inbetween

    domain_len = 1:length(domain);

    mask = ((domain_len > onsets(k)) .* (domain_len < offsets(k))).*domain_len;

    mask_nz = nonzeros(mask);

    rangeofvalsonset = [rangeofvalsonset,counts(mask_nz)];

    on_maxes = [on_maxes,max(rangeofvalsonset)];

end

%-----------------
% Check On Maxes
% Must be > 0.025 spikes/bins/trials

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

qual_height_on_maxes = on_maxes > min_max_height;

%-----------------
% Check On Maxes
% Must be > 3x mean preceeding 50 ms

% Find time of on_maxes ------> pretty close, needs fix to be exact

onset_domain = round(onsets);
onset_points = [];

for i = 1:length(onset_domain)

    onset_points(i,1) = domain(onset_domain(i));
    onset_points(i,2) = on_maxes(i);

end


% Go back 50ms from max to check mean

for i = 1:length(onset_points)

    prev_edge = onset_points(i,1) - 0.05;

    if (onset_points(i,1) - 0.05) < 0
        prev_edge = 0;
    end
    
    % find closest domain value to prev edge
    diffs = abs(domain - prev_edge);
    [min_diff, prev_index] = min(diffs);
    closest_domain_val = domain(prev_index);

    curr_index = find(domain==onset_points(i,1));

    prev_mean = mean(counts(prev_index:curr_index)); 

    if onset_points(i,2) >= 3*prev_mean
        disp('pass')
        qual_mean_on_maxes(i) = 1;
    else
        qual_mean_on_maxes(i) = 0;
    end

end

% Array of all onset points and if they qualify

qual_on_points = [onset_points(:,1),onset_points(:,2),...
    double(qual_height_on_maxes)',qual_mean_on_maxes'];

%%

%-----------------
% Offsets


    % for k = 1:length(offsets)
    % 
    %     rangeofvalsoffset = [];
    % 
    %     %Edge case 1
    %     s_offset2 = floor(offsets(k+flag));
    %     e_offset2 = ceil(offsets(k+flag));
    % 
    %     where_on_the_slope3 = offsets(k)-s_offset2;
    % 
    %     excat_val3 = (counts(e_offset2)-counts(s_offset2))*where_on_the_slope3  + counts(s_offset2);
    % 
    %     rangeofvalsoffset = [rangeofvalsoffset,excat_val3];
    % 
    % 
    %     %Edge case 2
    %     s_onset2 = floor(onsets(k));
    %     e_onset2 = ceil(onsets(k));
    % 
    % 
    %     where_on_the_slope4 = onsets(k)-s_onset2;
    % 
    %     excat_val4 = (counts(e_onset2)-counts(s_onset2))*where_on_the_slope4 + counts(s_onset2);
    % 
    %     rangeofvalsoffset = [rangeofvalsoffset,excat_val4];
    % 
    % 
    %     %Everything inbetween
    % 
    %     domain_len = 1:length(domain);
    % 
    %     mask2 = ((domain_len > offsets(k)) .* (domain_len < onsets(k))).*domain_len;
    % 
    %     mask_nz2 = nonzeros(mask2);
    % 
    %     rangeofvalsoffset = [rangeofvalsoffset,counts(mask_nz2)];
    % 
    %     off_maxes = [off_maxes,max(rangeofvalsoffset)];
    % 
    % end


%Todos

%Smooth out tiny little spikes that are hiding [+]
%Check on maxes [?]
%Get off maxes [-]








