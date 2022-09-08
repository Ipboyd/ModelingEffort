function plotUnitVoltage(varargin)

[y1,fs] = audioread('200k_target1.wav');
[y2] = audioread('200k_target2.wav');
yt = (0:length(y1)-1)/fs;

padToTime = 3.5;
y1 = [zeros(round(fs*0.05),1);y1];
y1 = [y1;zeros(fs*padToTime-length(y1),1)];

y2 = [zeros(round(fs*0.05),1);y2];
y2 = [y2;zeros(fs*padToTime-length(y2),1)];
yt = (0:length(y1)-1)/fs;

pop = varargin{1};
snn_out = varargin{2};

if nargin < 3
    varNum = 1;
else
    varNum = varargin{3};
end

t_vec = (0.1:0.1:3500)/1000 - 0.3;

% find cell voltage to be plotted
params = snn_out(varNum).model.parameters;
V_rest = params.([pop '_E_L']);

inputNames = {snn_out(varNum).model.specification.connections(strcmp({snn_out(varNum).model.specification.connections.target},pop)).source};
% inputSpks = cellfun(@(x) [x '_V_spikes'],inputNames,'uniformoutput',false);

if ~strcmp(pop,'On') && ~strcmp(pop,'Off')
    inputSpks = cellfun(@(x) [pop '_' x '_PSC_syn'],inputNames,'uniformoutput',false);
else
    inputSpks = cellfun(@(x) [pop '_' x '_IC'],inputNames,'uniformoutput',false);
end

% replace Poisson noise unit stuff
if any(strcmp(inputNames,pop)) && ~contains(pop,'IC')
    inputSpks{strcmp(inputNames,pop)} = [pop '_' pop '_iNoise_V3_sn'];
end

taus = zeros(size(inputNames));

flag = 0;
for s = 1:length(inputNames)
    
    if (strcmp(inputNames{s}(1),'X') && strcmp(pop(1),'R')) || strcmp(pop(1),'X')
        labels{s} = 'F'; flag = 1;
    elseif (strcmp(inputNames{s}(1),'S') && strcmp(pop,'C')) || ...
            (strcmp(inputNames{s}(1),'S') && strcmp(pop(1),'R')) || strcmp(pop(1),'S')
        labels{s} = 'P'; flag = 1;
    end
    
    if flag
        taus(s) = params.([pop '_' inputNames{s} '_PSC_tau' labels{s}]);
        flag = 0;
    end
    
end

nCells = size(snn_out(varNum).([pop '_V']),2);
targets = {snn_out(1).model.specification.connections.target};
sources = {snn_out(1).model.specification.connections.source};

% find index for input netcon
% if strcmp(pop(1),'R') %|| strcmp(pop(1),'S')
%     ind = find(strcmp(sources,['X' pop(2:end)]) & strcmp(targets,pop));
%     netParams = snn_out(1).model.specification.connections(ind).parameters;
% end

y_target = 0.2;
x0 = 0.09; y0 = 0.11;
xl = 0.7; yl = 0.2;
yvolt = 0.4;

for ch = 1:nCells % go through each channel
    
    figure('unit','inches','position',[3 3 9 5]); clf; hold on; 
    
    % plot target
    subplot('position',[x0 y0+yl+yvolt xl y_target]);

    if varNum < 11
        y_stim = y1;
    else
        y_stim = y2;
    end
    plot(yt,y_stim,'k'); xlim([yt(1) yt(end)]); set(gca,'ytick',[],'xtick',[]);

    if strcmp(pop,'C')
        title([pop ' voltage with input spks']);
    else
        title([pop ' channel ' num2str(ch) ' voltage with input spks']);
    end

    % add spikes to voltage trace
    V_trace = snn_out(varNum).([pop '_V'])(:,ch);
    V_trace(snn_out(varNum).([pop '_V_spikes'])(:,ch) == 1,ch) = -20;

    % plot voltage trace
    subplot('position',[x0 y0+yl xl yvolt]);
    plot(t_vec,V_trace); box off
    xlim([0 3500]/1000-0.3); ylabel('Voltage [mV]');
    set(gca,'xtick',[]);
        
    i = []; ct = 0;
    
    subplot('position',[x0 y0 xl yl]);
    hold on;
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
        
        % plot noise inputs
        if strcmp(inputNames{s},pop)
            ct = ct + 1;

            maxVal = max(abs(snn_out(varNum).(inputSpks{s})),[],'all');

            plot(t_vec,(snn_out(varNum).(inputSpks{s})(:,nc)/maxVal-ct)+V_rest);

            if taus(s) ~= 0
                line([420 420+taus(s)]/1000-0.3,[1 1]*(V_rest - ct),'linewidth',5,'color','k');
            end

            i = cat(2,i,s);
            continue
        end
        
        % plot inputs from other cells
        if any(params.([pop '_' inputNames{s} '_PSC_gSYN']))
            for nc = 1:nInputCells % input channels
                if netcon(nc,ch) % if the input synapses onto pop
                    ct = ct + 1;

                    maxVal = max(abs(snn_out(varNum).(inputSpks{s})),[],'all');
                    
                    plot(t_vec,(-snn_out(varNum).(inputSpks{s})(:,nc)/maxVal-ct)+V_rest);
                    
                    if taus(s) ~= 0
                        line([420 420+taus(s)]/1000-0.3,[1 1]*(V_rest - ct),'linewidth',5,'color','k');
                    end
                    
                    i = cat(2,i,s);
                end
            end
        end
    end
    % line([420 420 + tau],[1 1]*(V_rest - ct-0.5),'linewidth',5,'color','k');
    
    % adaptation
    if any(snn_out(varNum).([pop '_g_ad']))
        maxVal = max(abs(snn_out(varNum).([pop '_g_ad'])),[],'all');
        plot(t_vec,(-snn_out(varNum).([pop '_g_ad'])(:,nc)/maxVal-ct)+V_rest);

        tau_ad = params.([pop '_tau_ad']);
        line([420 420+tau_ad]/1000-0.3,[1 1]*(V_rest - ct),'linewidth',5,'color','k');
    end

    legdata = cell(0);
    % go thru inputs to make legend
    inputCts = ones(1,max(i));
    for s = i

        % if unit is C, note which channel inputs come from
        if strcmp(pop,'C') && ~strcmp(inputNames{s},pop)
            legdata = cat(1,legdata,[inputNames{s} ' CH' num2str(inputCts(s)) ' spks']);
        elseif strcmp(inputNames{s},pop) % noise inputs
            legdata = cat(1,legdata,'Poisson noise');
        else
            legdata = cat(1,legdata,[inputNames{s} ' PSC']);
        end
        if taus(s) ~= 0
            legdata = cat(1,legdata,['\tau_{' labels{s} '} (' num2str(taus(s)) 'ms)']);
        end
        inputCts(s) = inputCts(s) + 1;
    end

    if any(snn_out(varNum).([pop '_g_ad']))
        legdata = cat(1,legdata,'SR adaptation',['\tau_{ad} (' num2str(tau_ad) 'ms)']);
    end
    legend(legdata,'position',[0.80 0.15 0.15 0.15]);

end
xlim([0 3500]/1000-0.3);
xlabel('Time (s)'); ylabel('Inputs'); set(gca,'ytick',[],'fontsize',10);

end