close all

cd(userpath);
cd('../GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting')

load('all_units_info_with_polished_criteria_modified_perf.mat','all_data');
load('sound_files.mat','sampleRate','target1','target2');  %Sample Rate 195312 Hz


cd('PicturesToFit\')
%Look at the first animal type
for n = 1
    
    SpikeTimes = all_data(n).ctrl_tar1_timestamps(:,1);
    
    %The stim lasts from 0s to 2.9801 seconds
    %We are going to run the sim from 0 to 2.9801
    %We need to extract all values between this for the spike distnace loss
    %measures.
    %Switch to zeros for spy plot
    picture = zeros(10,29801);
    
    for m = 1:10
        stim_mask = logical((SpikeTimes{m} > 0) .* (SpikeTimes{m} < 2.9801));
        trial_indicies = round(SpikeTimes{m}(stim_mask)*10000);
        %Switch to one for spy plot
        %1 added for zero indexing. Lets say for instance a spike lands at
        %time = 0. In matlab this would be at the first indicy in the
        %picture which is indicy 1.
        picture(m,trial_indicies+1) = 1;
    end
    figure;
    subplot(3,1,1)
    spy(picture);
    subplot(3,1,2)
    plot(target1)
    subplot(3,1,3)
    picture2 = picture.*(1:length(picture));
    histogram(nonzeros(picture2),300)
    
    %save('picture_fit'+  string(n) + 'contra' +'.mat','picture');
end

%picture = uint8(picture*255);
%picture = repmat(picture, 1, 1, 3);


%imwrite(picture, 'raster_data4col1.png'); 


%Figure out how much silence there is at the input relative to the amount
%of silence for the stimuli

%for m = 1:10

%end
