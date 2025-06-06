%Pad the end for now. It looks like the way that we set things up
%everything is off by 1 index?


pre_list = x{1};
spks_store = pre_list{1};

titles = {'On','Off','R1On','R2On','S2OnOff','S1OnOff','R2Off','R1Off'};


for m = 1:length(spks_store)

    num_epochs = 1;
    
    spks_return = spks_store{m};
    
    formatted_spike_trains = {};
    
    for k = 1:num_epochs
        
        %Change back after running epochs ################
        cur_epoch = spks_store{m};
        cur_epoch_len = length(spks_store{m});
        %cur_epoch = spks_return{k};
        %#cur_epoch_len = length(spks_return{k});
    
        matlab_holder = zeros(1,cur_epoch_len);
    
        
    
        for j = 1:cur_epoch_len
            %matlab_holder(j) = single(cur_epoch{j});
            matlab_holder(j) = single(cur_epoch{j});
        end

        sum(matlab_holder)
    
        matlab_holder_reshaped = reshape(matlab_holder,34998,11);
    
        formatted_spike_trains{end+1} = matlab_holder_reshaped;
    
    end
    
    figure(Position=[600,200,900,750]);
    
    
    frs = [];
    
    for j = 1:num_epochs
        plotting_data = [];
    
        %subplot(mod(j,5),ceil(j/(num_epochs/2)),j)
    
        %Song 1
    
        for k = 1:10
            
            spike_epoch = formatted_spike_trains{j};
            R2on = spike_epoch(:,k+1); %Doing +1 because we currently return an extra row of 0s due to the way that the spikes are initialized.
    
            reference_cell = R2on;
    
            times = find(reference_cell == 1);
            plotting_data = [plotting_data;times];
    
        end
    
        frs = [frs,sum(sum(plotting_data))];
    
        raw_freq_data = histcounts(plotting_data, BinWidth=200);
        avg_data1 = movmean(raw_freq_data,3);
        
        avg_data = avg_data1/10/0.02*(10/150); %Divide by 10 for # of trials, divide by 0.02 (bindWidth) to get spikes/s = Hz,next part is scale factor which will be reprented with scaling bar in paper
        plot(avg_data,LineWidth=1.5); hold on
        yticklabels([0:75:150]);
        xticklabels('')
    
    end

    title(titles{m});

end

figure();
plot(frs);

% Return this next time
%loss = 
