cd(userpath);
cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')

clear all
close all

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz


SpikeTimes = all_data(7).ctrl_tar1_timestamps(:,1);

%The stim lasts from 0s to 2.9801 seconds.
%We are going to run the sim from 0 to 2.9801
%We need to extract all values between this for the spike distnace loss
%measures.
picture = zeros(10,29801);

for m = 1:10
    stim_mask = logical((SpikeTimes{m} > 0) .* (SpikeTimes{m} < 2.9801));
    trial_indicies = round(SpikeTimes{m}(stim_mask)*10000);
    picture(m,trial_indicies) = 1;
end

spy(picture);

save('picture_fit.mat','picture');



%Figure out how much silence there is at the input relative to the amount
%of silence for the stimuli

%for m = 1:10

%end
