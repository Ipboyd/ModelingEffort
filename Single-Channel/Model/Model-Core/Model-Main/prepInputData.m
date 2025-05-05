% locs are based on the locations for stimuli in mouse experiments
% +90, 45, and 0 are to the left (contra)
% -90 to the right (ipsi)
locs = [90 45 0 -90];

% only create spks file if not done yet

tmax = max(cellfun(@numel,fr_target_on));
labels = {'on','off'};
for ICtype = [1 2]
    % divide all times by dt to upsample the time axis
    spks = [];
    % spks = gpuArray(spks);

    % load fr traces, scale by weights

    

    for z = subz
        
        %Removed locs so that we can use the incex of spatial curves to
        %just index those 4 points. 9/3 IB

        % target and masker weights
        if z <= 4  % masker only
            flag = 1;
            tloc(z) = nan;

            mloc(z) = z;
        elseif mod(z,5) == 0 % target only
            
            flag = 0;
            tloc(z) = floor(z/5);
            
            mloc(z) = nan;
        else % mixed
            flag = 1;
            tloc(z) = floor(z/5);
            mloc(z) = mod(z,5);
        end

        singleConfigSpks = zeros(20,1,tmax);
        % singleConfigSpks = gpuArray(singleConfigSpks);

        for t = 1:20 % trials: first 10 trials are target 1, last 10 are target 2
            for ch = 1:options.nCells
                
                %Added 8/30 as a request from Dr. Sen to show a more
                %realstic input according to our data.
                %Changed on 9/3 to just respond to the curves accordingly
                
                if flag == 1

                    if isnan(tloc(z))
                        t_wt = 0;
                        m_wt = masked_spatialCurves(ch,mloc(z));
                    else
                        t_wt = masked_spatialCurves(ch,tloc(z));
                        m_wt = masked_spatialCurves(ch,mloc(z));
                    end
                
                else

                    if isnan(mloc(z))
                        t_wt = spatialCurves(ch,tloc(z));
                        m_wt = 0;
                    else
                        t_wt = spatialCurves(ch,tloc(z));
                        m_wt = spatialCurves(ch,mloc(z));
                    end

                    
                end
                
                
                % if toggle_real == 1
                %     if flag == 1
                %         t_wt = masked_spatialCurves(ch,azi == tloc(z));
                %         m_wt = masked_spatialCurves(ch,azi == mloc(z));
                %     else
                %         t_wt = spatialCurves(ch,azi == tloc(z));
                %         m_wt = spatialCurves(ch,azi == mloc(z));
                %     end
                % else
                %         t_wt = spatialCurves(ch,azi == tloc(z));
                %         m_wt = spatialCurves(ch,azi == mloc(z));
                % end
                
                
                %IB 7/15 decreased the M-weight for sanity check


                %m_wt = spatialCurves(ch,azi == mloc(z))*1/100;
                

                if isempty(t_wt), t_wt = 0; end
                if isempty(m_wt), m_wt = 0; end

                if t <= 10 %song 1
                    %singleConfigSpks(t,ch,:) = t_wt.*eval(['fr_target_' labels{ICtype} '{1}']) + m_wt.*fr_masker{t};
                    t_wt = 1;
                    singleConfigSpks(t,ch,:) = eval(['fr_target_' labels{ICtype} '{1}']);
                else
                    t_wt = 1;
                    %singleConfigSpks(t,ch,:) = t_wt.*eval(['fr_target_' labels{ICtype} '{2}']) + m_wt.*fr_masker{t-10};
                    singleConfigSpks(t,ch,:) = eval(['fr_target_' labels{ICtype} '{2}']);
                end

                % if t_wt + m_wt >= 1
                %     singleConfigSpks(t,ch,:) = singleConfigSpks(t,ch,:) / (t_wt + m_wt);
                % end
            end
        end

        % format of spks is : [trial x channel x time]
        % pad each trial to have duration of padToTime
        if size(singleConfigSpks,3) < padToTime/dt
            padSize = padToTime/dt-size(singleConfigSpks,3);
            singleConfigSpks = cat(3,singleConfigSpks,zeros(20,options.nCells,padSize));
        end

        spkies = singleConfigSpks;
        
        %Current hypothesis is that snn contains 4 sets seperated by 7000

        % increase or decrease gain factor of strf
        % spks = cat(3,spks,singleConfigSpks(:,1,:)) * newStrfGain / strfGain;
        spks = cat(3,spks,singleConfigSpks) * newStrfGain / strfGain;
        % strfGain is from default_STRF_with_offset_200k.mat == 0.1
        spkies2 = squeeze(spks(1,:,:));

    end
    % format of spks should be : [time x channel x trial]
    spks = permute(spks,[3 2 1]);
    
    %study_dir2 = 'C:\Users\Admin\Documents\GitHub\MouseSpatialGrid';

    %save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks');
    %save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.dat']),'spks');
    save(fullfile(study_dir, 'solve',['IC_spks_' labels{ICtype} '.mat']),'spks','dt');
end
