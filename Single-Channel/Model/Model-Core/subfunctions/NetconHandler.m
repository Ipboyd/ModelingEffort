% NETCONS enable connectivity between different channels of the network
% row = source, column = target 
netcons = struct; 

% all other netcons defined as identity in columnNetwork_simpler_onoff

% XRnetcon: SOM (X) -> E (Ron)
netcons.XRnetcon = [0,1,1,1;
                    1,0,1,1;
                    1,1,0,1;
                    1,1,1,0];

% PEnetcon: PV (SOnOff) -> E (Ron)
%netcons.PEnetcon = eye(options.nCells);
netcons.PEnetcon = [1,1,1,1;
                    1,1,1,1;
                    1,1,1,1;
                    1,1,1,1];

% RCnetcon: E (Ron) -> C
netcons.RCnetcon = [1;1;1;1];


% netcons.CrossStatnetcon = [0,0,0,0;
%                           0,0,0,0;
%                           0,0,0,0;
%                           0,0,0,0];
