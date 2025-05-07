options.strfGain = strfGain;
options.dt = dt;

if isempty(options.locNum), options.time_end = size(spks,1)*dt; % [ms];
else, options.time_end = padToTime*numel(options.locNum); end