function plotUnitVoltage(varargin)

pop = varargin{1};
snn_out = varargin{2};

if nargin < 3
    varNum = 1;
else
    varNum = varargin{3};
end

t_vec = 0.1:0.1:3500;

% find cell voltage to be plotted
params = snn_out(varNum).model.parameters;
V_rest = params.([pop '_E_L']);

inputNames = {snn_out(varNum).model.specification.connections(strcmp({snn_out(varNum).model.specification.connections.target},pop)).source};
inputSpks = cellfun(@(x) [x '_V_spikes'],inputNames,'uniformoutput',false);

taus = zeros(size(inputNames));

flag = 0;
for s = 1:length(inputNames)
    
    if (strcmp(inputNames{s}(1),'X') && strcmp(pop(1),'R')) || strcmp(pop(1),'X'), labels{s} = 'F'; flag = 1;
    elseif (strcmp(inputNames{s}(1),'S') && strcmp(pop,'C')) || ...
            (strcmp(inputNames{s}(1),'S') && strcmp(pop(1),'R')) || strcmp(pop(1),'S'), labels{s} = 'P'; flag = 1; end
    
    if flag
        taus(s) = params.([pop '_' inputNames{s} '_iPSC_LTP_tau' labels{s}]);
        flag = 0;
    end
    
end

nCells = size(snn_out(varNum).([pop '_V']),2);
targets = {snn_out(1).model.specification.connections.target};
sources = {snn_out(1).model.specification.connections.source};

% find index for input netcon
if strcmp(pop(1),'R')
    ind = find(strcmp(sources,['X' pop(2:end)]) & strcmp(targets,pop));
    netParams = snn_out(1).model.specification.connections(ind).parameters;
end

for ch = 1:nCells % go through each channel
    
    figure; clf; hold on;
    
    plot(t_vec,snn_out(varNum).([pop '_V'])(:,ch));
    
    if strcmp(pop,'C')
        title([pop ' voltage with input spks']);
    else
        title([pop ' channel ' num2str(ch) ' voltage with input spks']);
    end
    
    xlabel('Time (ms)')
    
    i = []; ct = 0;
    
    % plot input spikes below voltage trace
    for s = 1:length(inputSpks)
        
        nInputCells = size(snn_out(varNum).(inputSpks{s}),2);
        
        % find input netcon if input is X cell
        if strcmp(inputNames{s}(1),'X')
            netcon = netParams{find(strcmp(netParams,'netcon'))+1};
        elseif strcmp(pop,'C')
            netcon = ones(nInputCells,1);
        else
            netcon = eye(nCells);
        end
        
        if strcmp(inputNames{s},pop), continue, end
        
        if any(params.([pop '_' inputNames{s} '_iPSC_LTP_gSYN']))
            for nc = 1:nInputCells % input channels
                if netcon(nc,ch) % if the input synapses onto pop
                    ct = ct + 1;
                    
                    plot(t_vec,(snn_out(varNum).(inputSpks{s})(:,nc)-ct)+V_rest);
                    
                    if taus(s) ~= 0
                        line([420 420+taus(s)],[1 1]*(V_rest - ct),'linewidth',5,'color','k');
                    end
                    
                    i = cat(2,i,s);
                end
            end
        end
    end
    % line([420 420 + tau],[1 1]*(V_rest - ct-0.5),'linewidth',5,'color','k');
    
    legdata = cell(0);
    % go thru inputs
    inputCts = ones(1,max(i));
    for s = i
        if strcmp(pop,'C')
            legdata = cat(1,legdata,[inputNames{s} ' CH' num2str(inputCts(s)) ' spks']);
        else
            legdata = cat(1,legdata,[inputNames{s} ' spks']);
        end
        if taus(s) ~= 0
            legdata = cat(1,legdata,['\tau_{' labels{s} '} (' num2str(taus(s)) 'ms)']);
        end
        inputCts(s) = inputCts(s) + 1;
    end
    legend([pop;legdata]);
end

end