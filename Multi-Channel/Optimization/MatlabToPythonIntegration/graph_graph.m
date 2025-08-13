% Example data
s = [1 1 3 3 2 2 4 4 6 5];   % source nodes
t = [4 3 4 5 5 3 7 6 7 6];   % target nodes
weights = coeff(:,10);  % your 10 edge weights

% Define node names (must be a cell array of char vectors or string array)
nodeNames = ["On","Off","S1","R1On","R1Off","S2","R2"];

% Desired positions (column-like “onset / PV / offset” layout)
pos = table( ...
   ["On";   "S1";  "Off";  "R1On"; "R1Off"; "S2";  "R2"], ... % <- node names
   [0;      1.0;     2.0;      0;      2.0;       1;     0], ...   % X
   [0.0;    1.0;   0.0;    2.0;    2.0;     3.0;   4.0], ... % Y
   'VariableNames', {'Name','X','Y'});



% Create graph using node names instead of numeric IDs
G = digraph(nodeNames(s), nodeNames(t), weights);


% Plot with edge thickness proportional to weight
figure;
h = plot(G, 'Layout', 'force');
%h.LineWidth = 1 + 3*(G.Edges.Weight / max(G.Edges.Weight)); % scale thickness
%h.EdgeCData = G.Edges.Weight;  % color edges by weight

% Map node names to our coordinates
[tf, loc] = ismember(string(G.Nodes.Name), pos.Name);
assert(all(tf), 'Some nodes are missing from the position table.');
h.XData = pos.X(loc);
h.YData = pos.Y(loc);


% Edge thickness proportional to magnitude of weight
h.LineWidth = 1 + 10*(abs(G.Edges.Weight) / max(abs(G.Edges.Weight)));

% Edge colors based on signed weight
h.EdgeCData = G.Edges.Weight;

colormap parula; colorbar;
title('Weighted Graph');


% --- your graph (directed or undirected) ---
% e.g.: G = digraph(s, t, w, nodeNames);






% If you already colored/thickened edges by weight, keep those lines
% e.g.:
% h.LineWidth = 1 + 4*(abs(G.Edges.Weight)/max(abs(G.Edges.Weight)));
% h.EdgeCData = G.Edges.Weight; colormap(turbo); colorbar; caxis([-1 1]);