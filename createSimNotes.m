function annotParams = createSimNotes(snn_out,expName,model)
% create txt file with information on all simulations in experiment

varied_params = snn_out(1).varied;

if strcmp(model.type,'On') || strcmp(model.type,'Off') 
varied_params(1) = [];   % if varies(1) includes On ad Off
else
varied_params(1:2) = [];   % On and Off
end

numVaries = length(snn_out)/20;

fid = fopen(fullfile('simData',expName, 'notes.txt'), 'w');
if fid == -1
   error('Cannot open log file.'); 
end

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

feat_interaction = {'No','Yes'};

fprintf(fid,['Model type: ' model.type ', Interaction: ' feat_interaction{model.interaction + 1}  '\n\n']);

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

