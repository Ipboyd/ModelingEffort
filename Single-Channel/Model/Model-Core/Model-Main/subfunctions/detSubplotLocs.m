function [subplot_locs,n_rows,n_cols] = detSubplotLocs(popNames)

% determines subplot locations based on cells present

% Check if X cells are included
hasX = any(contains(popNames,'X'));

% Check if there's only one population of PV neurons per layer (will be
% named S, not S1On/S1Off/etc.)
hasPVconverge = ~any(contains(popNames,'S') & ~contains(popNames,'On'));

n_rows = 3;
n_cols = 4 + hasX;

% topPops = {'C','R2On','R2Off','TD','S2On','S2Off','S2'};
% middlePops = {'S','S1','S1On','S1Off','R1On','R1Off','ROn','ROff','SOn','SOff','X'};
% bottomPops = {'On','Off'};

topPops = {'C','R2On','R2Off','TD','S2OnOff'};
middlePops = {'S1OnOff','R1On','R1Off','ROn','ROff','X'};
bottomPops = {'On','Off'};

% initalize unit subplot locations by layer
subplot_locs(ismember(popNames,bottomPops)) = (n_rows-1)*n_cols;
subplot_locs(ismember(popNames,topPops)) = 0;
subplot_locs(ismember(popNames,middlePops)) = (n_rows-2)*n_cols;

% all On neurons (including C) will be on the first and second columns
subplot_locs(contains(popNames,["On","C"])) = subplot_locs(contains(popNames,["On","C"])) + 2;
% PV-On should be on the first column
subplot_locs(contains(popNames,'On') & contains(popNames,'S')) = subplot_locs(contains(popNames,'On') & contains(popNames,'S')) - 1;

% Off columns should be on the third and fourth columns
subplot_locs(contains(popNames,'Off')) = subplot_locs(contains(popNames,'Off')) + n_cols;
% PV-off should be on the third
subplot_locs(contains(popNames,'Off') & contains(popNames,'S')) = subplot_locs(contains(popNames,'Off') & contains(popNames,'S')) - 1;

% if X and TD cells are in circuit, they should be in the middle, between the on
% and off columns
subplot_locs(contains(popNames,["X","TD"])) = subplot_locs(contains(popNames,["X","TD"])) + 3;

% If there's only one PV neuron per layer, shift each excitatory neuron location 1 to
% the left and shift the single PV to the middle column

% ('S' doesn't fulfill any of the PV subplot_loc conditions above)
if hasPVconverge
    subplot_locs(contains(popNames,["On","C"])) = subplot_locs(contains(popNames,["On","C"])) - 1;
    subplot_locs(contains(popNames,'Off')) = subplot_locs(contains(popNames,'Off')) - 1;
    subplot_locs(contains(popNames,'S') & ~(contains(popNames,["On","Off"]) )) = ...
        subplot_locs(contains(popNames,'S') & ~(contains(popNames,["On","Off"]))) + 2;

    subplot_locs(contains(popNames,["X","TD"])) = subplot_locs(contains(popNames,["X","TD"])) - 1;
end

end
