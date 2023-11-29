function annotParams = createSimNotes(snn_out,simDataDir,options)
% create txt file with information on all simulations in experiment

varied_params = snn_out(1).varied;
varied_params(1) = [];

numVaries = length(snn_out)/20;

fid = fopen(fullfile(simDataDir, 'notes.txt'), 'w');
if fid == -1
   error('Cannot open log file.'); 
end

fprintf(fid, sprintf('STRF gain: %f \n\n',options.strfGain));

% find the params that vary between simulations; don't show parameters that
% stay the same on the grids
if numVaries > 1
    for p = 1:length(varied_params)
        temp(p,:) = [snn_out(1:numVaries).(varied_params{p})];
    end
    annotInds = find(~all(temp == temp(:,1),2));
else
    annotInds = 1:length(varied_params);
end

% % write down source of IC inputs
% ICdir = strsplit(options.ICdir,'\'); ICdir = join(ICdir,'\\');
% fprintf(fid,['IC spikes from: ' ICdir{1} '\n\n']);

for vv = 1:numVaries
    
    fprintf(fid, ['Simulation #' num2str(vv) '\n']);
    
    annotStr = [];
    
    % concatenate varied params
    annotStr{1} = strcat(varied_params{1}, ' = ', num2str(snn_out(vv).(varied_params{1})));
    
    for f = 2:length(varied_params)
        annotStr{end+1} = strcat(varied_params{f}, ' = ', num2str(snn_out(vv).(varied_params{f})));
    end    
    
    for k = 1:length(annotStr), fprintf(fid, [annotStr{k} '\n']); end
    fprintf(fid,'\n');
    
    annotParams{vv} = annotStr(annotInds);
    
end

fclose(fid);

end

