% gsyn networks enable pathway specific synaptic strength modification
% row = source, column = target 
gsyncons = struct; 

% all other gsyn networks defined as identity in columnNetwork_simpler_onoff

% OnRgsyncon: Onset (On) -> ROn
% default was 0.02
%gsyncons.OnRgsyncon = [varied_struct.ONtoR1,varied_struct.ONtoR2,varied_struct.ONtoR3,varied_struct.ONtoR4];
%Pass through
gsyncons.OnRgsyncon = [0.022,0.022,0.022,0.022];


% XRgsyncon: SOM (X) -> E (Ron)
% gsyncons.XRgsyncon =   [0.000,varied_struct.XR1,varied_struct.XR2,varied_struct.XR3;
%                         varied_struct.XR4,0.000,varied_struct.XR5,varied_struct.XR6;
%                         varied_struct.XR7,varied_struct.XR8,0.000,varied_struct.XR9;
%                         varied_struct.XR10,varied_struct.XR11,varied_struct.XR12,0.000];
% gsyncons.XRgsyncon =   [0.000,0.000,0.000,0.000;
%                         0.000,0.000,0.000,0.000;
%                         0.000,0.000,0.000,0.000;
%                         0.000,0.000,0.000,0.000];

gsyncons.XRgsyncon =   [0.000,0.010,0.010,0.010;
                        0.000,0.000,0.000,0.000;
                        0.000,0.000,0.000,0.000;
                        0.000,0.000,0.000,0.000];

% RCgsyncon: E (Ron) -> C
% needs to be transposed for PSC3 mech
%Nominal
%gsyncons.RCgsyncon = [0.0088, 0.0043, 0.000466, 0.0060].';
%gsyncons.RCgsyncon = [varied_struct.RtoC1, varied_struct.RtoC2, varied_struct.RtoC3, varied_struct.RtoC4].';

gsyncons.RCgsyncon = [0.01, 0.01, 0.01, 0.01].';



% gsyncons.PEgsyncon = [varied_struct.IntraPV1,varied_struct.CrossPV1,varied_struct.CrossPV2,varied_struct.CrossPV3;
%                       varied_struct.CrossPV4,varied_struct.IntraPV2,varied_struct.CrossPV5,varied_struct.CrossPV6;
%                       varied_struct.CrossPV7,varied_struct.CrossPV8,varied_struct.IntraPV3,varied_struct.CrossPV9;
%                       varied_struct.CrossPV10,varied_struct.CrossPV11,varied_struct.CrossPV12,varied_struct.IntraPV4];

gsyncons.PEgsyncon = [0.02,0.000,0.000,0.000;
                      0.000,0.02,0.000,0.000;
                      0.000,0.000,0.02,0.000;
                      0.000,0.000,0.000,0.02];


% gsyncons.CrossStatgsyncon = [0.000,0.000,0.000,0.000;
%                               0.000,0.000,0.000,0.000;
%                               0.000,0.000,0.000,0.000;
%                               0.000,0.000,0.000,0.000];

%Nominally 0.025 for pvs
