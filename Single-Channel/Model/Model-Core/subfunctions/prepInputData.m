%padToTime = 3500; % [ms]
padToTime = 2980.1;
tmax = max(cellfun(@numel,fr_target_on));
labels = {'on','off'};
for ICtype = [1 2]
    % divide all times by dt to upsample the time axis
    spks = [];
    singleConfigSpks = zeros(20,1,tmax);
    for t = 1:20
        if t <= 10 %song 1
            singleConfigSpks(t,1,:) = eval(['fr_target_' labels{ICtype} '{1}']);
        else
            singleConfigSpks(t,1,:) = eval(['fr_target_' labels{ICtype} '{2}']);
        end
    end
    if size(singleConfigSpks,3) < padToTime/dt
        padSize = padToTime/dt-size(singleConfigSpks,3);
        singleConfigSpks = cat(3,singleConfigSpks,zeros(20,options.nCells,padSize));
    end

    spkies = singleConfigSpks;
    spks = cat(3,spks,singleConfigSpks);
    spkies2 = squeeze(spks(1,:,:));


    spks = permute(spks,[3 2 1]);

    figure;
    plot(eval(['fr_target_' labels{ICtype} '{2}']))
    

    save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks','dt');
end
