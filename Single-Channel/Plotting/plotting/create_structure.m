% run either plot_digraph or plotAll_digraphs after running this for
% graphing data
% load('current_run_data.mat');

%update netcons
NetconHandler;

%Number of nodes (x data)
%heirarchy of nodes (y data)
channels = 1:4;

numChannels = length(channels);
numPopulations = length(s.populations);
numConnections = length(s.connections);

% population without C
le = numPopulations-1;

% Total # of nodes, channels*(population without C) plus 1 for output C
totalNodes = (numChannels*le)+1;

% Preallocate for efficiency
Nodesx = zeros(1,totalNodes);
Nodesy = zeros(1,totalNodes);

% node counter as index
nodeCounter = 0;

%% Mapping of population names to x and y offsets
popNameOffsets = containers.Map(...
    {'On', 'Off', 'ROn', 'ROff', 'SOnOff', 'TD', 'X'}, ...
    {[0, 0], [0.5, 0], [0, 2], [0.5, 2], [0.25, 1], [0.75, 3], [-0.25, 1]});



for b = 1:numChannels
    for a = 1:numPopulations
        popName = s.populations(a).name;

        if isKey(popNameOffsets, popName)
            offsets = popNameOffsets(popName);
            nodeCounter = nodeCounter + 1;
            Nodesx(nodeCounter) = channels(b) + offsets(1);
            Nodesy(nodeCounter) = offsets(2);
        end

    end
end

%% single offset implementation for C
nodeCounter = nodeCounter + 1;
Nodesx(nodeCounter) = (numChannels+1)/2 + .25;
Nodesy(nodeCounter) = 3.5;


%% Determining the start and end node #'s
%Should be able to somehow automatically populate this with s.connections
sources = [];
targets = [];

inhibs_PEs = [];
inhibs_PEt = {};

inhibs_XRs = [];
inhibs_XRt = [];
SOM_nodes = [];

TD_SOMs = [];
TD_SOMt = [];
[XR_m,XR_n] = size(netcons.XRnetcon);


%Two sanity checks
%If only connect contra make sure c looks like contra
%All connections rerun on normal tuning curves

%% create graphable arrays of start and end node locations for each connection
cons_possible = {'On->On', 'Off->Off', 'On->ROn', ...
                 'On->SOnOff', 'SOnOff->Ron', 'SOnOff->ROff', ...
                 'Off->ROff', 'Off->SOnOff', 'ROn->ROn', ...
                 'ROn->X', 'X->ROn', 'TD->ROn', ...
                 'TD->ROff', 'TD->X', 'ROn->C'};

inhibitory_cons = {'SOnOff', 'X', 'TD'};

% mapping for node numberings
nodeMap = containers.Map({'On', 'Off', 'ROn', 'ROff', 'SOnOff', 'TD', 'X', 'C'}, ...
                           {1, 2, 3, 4, 5, 6, 7, totalNodes});

% reverse mapping
reverseNodeMap = containers.Map(values(nodeMap), keys(nodeMap));


%Current node graph does not handle C nodes?

for c = 1:numChannels
    for x = 1:numConnections
        % grab direction string
        direction = s.connections(x).direction;

        % using current netcon should make this dynamic
        netcon_index = find(strcmp('netcon', s.connections(x).parameters)) + 1;
        current_netcon = s.connections(x).parameters(netcon_index);
        current_netcon = current_netcon{1};
        [m,n] = size(current_netcon);
        
        % get start node and end node
        str = extractBefore(direction,"-");
        stra = extractAfter(direction,">");

        % we don't want On->On, or Off->Off
        if strcmp(str,stra)
            continue;
        end

        % Check if the connection is inhibitory
        if ismember(str, inhibitory_cons)
            for p = 1:n
                if current_netcon(c, p) == 1
                    s_inhib = (c-1) * le + nodeMap(str);
                    t_inhib = (p-1) * le + nodeMap(stra);
                    if strcmp(str, 'X')
                        inhibs_XRt = [inhibs_XRt t_inhib];
                        inhibs_XRs = [inhibs_XRs s_inhib];
                        SOM_nodes = [SOM_nodes s_inhib];
                    elseif strcmp(str, 'TD') && strcmp(stra, 'X')
                        TD_SOMs = [TD_SOMs s_inhib];
                        TD_SOMt = [TD_SOMt t_inhib];
                        
                        % if no outgoing connection from SOM, add end node
                        % loc to SOM node array for red coloring
                        SOM_nodes = [SOM_nodes t_inhib];
                    elseif strcmp(str, 'SOnOff')
                        if c==p % within channel, flag 1
                            inhibs_PEt{end+1} = {t_inhib, 1};
                        else    % cross channel, flag 2
                            inhibs_PEt{end+1} = {t_inhib, 2};
                        end
                        inhibs_PEs = [inhibs_PEs s_inhib];
                    end
                    targets = [targets t_inhib];
                    sources = [sources s_inhib];
                end
            end
        else
            % Handle non-inhibitory connections
            for p = 1:n
                if current_netcon(c, p) == 1
                    % assign source node a number
                    source = (c-1) * le + nodeMap(str);

                    % if target node is C, label as last node
                    if strcmp(stra, 'C')
                        tar = nodeMap(stra);
                    else
                        % assign target node a number
                        tar = (p-1) * le + nodeMap(stra);
                    end
                    % add assigned node # to source and target arrays
                    sources = [sources source];
                    targets = [targets tar];
                end
            end
        end
    end
end


%% synaptic strength modifier
%Now we have to go through and "highlight" which means to change the size
%of the line widths that connect neurons.
syns = [];
all_gsyns = [];
arrow_sizes = [];

%Might have to do this for inhibs_2/s2
for u = 1:length(sources)
    start_type = mod(sources(u),7);
    end_type = mod(targets(u),7);

    if start_type == 0
        start_type = 7;
    end
    if end_type == 0
        end_type = 7;
    end

    %This might be one way to handle it. Assuming we just have the one
    %cortical netcon and it is always the last one.
    if(targets(u) == nodeCounter)
        end_type = 8;
    end

    tstring = s.populations(start_type).name + "->" + s.populations(end_type).name;
    for j = 1:length(s.connections)
        if strcmp(tstring,s.connections(j).direction)
            target = j;
        end
    end

    gsyn_m = s.connections(target).parameters(2);

    % check to see if we are working with a netcon of gsyn values
    % (different for each connection)
    if size(gsyn_m{1}) == [4,4]
        gsyn_net = gsyn_m{1};
        row_num = floor((sources(u) - 1) / 7) + 1;
        col_num = floor((targets(u) - 1) / 7) + 1;
        gsyn = gsyn_net(row_num, col_num);
        % if X or PV and ending on ROn
        if (start_type == 5 || start_type == 7) && end_type == 3
            if gsyn > 0.2 
                end_gsyn = 1;
            else
                % end_gsyn = (gsyn^1.5)*750;
                end_gsyn = gsyn*400;
                arrow_size = end_gsyn*3;
            end
        else
            end_gsyn = 1;
            arrow_size = end_gsyn*6;
        end
    % cross channel PV condition
    elseif start_type == 5 && (sources(u) - targets(u) ~= 2) && end_type == 3
            gsyn = gsyn_m{1};
            end_gsyn = gsyn*100;
            arrow_size = end_gsyn*3;
    else
        end_gsyn = 1;
        arrow_size = end_gsyn*6;
    end
    % elseif gsyn_m{1} > 0.2
    %     end_gsyn = 0.5;
    % else
    %     end_gsyn = (gsyn_m{1}^1.5)*750;
    % end


    syns = [syns gsyn_m];

    all_gsyns = [all_gsyns end_gsyn];

    arrow_sizes = [arrow_sizes arrow_size];

end
persistent_gui(TD_SOMs, TD_SOMt, inhibs_XRs, inhibs_XRt, inhibs_PEs, inhibs_PEt, targets, sources, Nodesx, Nodesy, SOM_nodes, all_gsyns, arrow_sizes, reverseNodeMap);