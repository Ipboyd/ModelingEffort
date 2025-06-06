% firing rates were generated with sampling rate of 10000 Hz to match old
% simulation time step, downsample if dt's don't match
if dt ~= 0.1
    dsamp_fac = dt/0.1;
    %for m = 1:10    
    %    fr_masker{m} = downsample(fr_masker{m},dsamp_fac);
    %end
    for t = 1:2
        fr_target_on{t} = downsample(fr_target_on{t},dsamp_fac);
        fr_target_off{t} = downsample(fr_target_off{t},dsamp_fac);
    end
end