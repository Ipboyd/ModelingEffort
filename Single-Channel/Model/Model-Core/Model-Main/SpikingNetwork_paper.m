%                                                                   %
% The following Script acts a "main" for the single channel effort. %
%                                                                   %

%% Initialize

% change current directory to folder where this script is stored
mfileinfo = mfilename('fullpath');
mfiledir = strsplit(mfileinfo,filesep);

%Create paths to all peripheral scripts
addpath('../../Neuron Modeling/mechs');                                             %LIF equations for neurons
addpath(genpath('../../PreCortical-Modeling'));                                     %pre-cortical modeling  
addpath('../../Peripherals/cSPIKE');
addpath('../../Peripherals/cSPIKE/cSPIKEmex'); InitializecSPIKE;                    %Spike Distance calculator
addpath('../subfunctions');                                                         %assistive functions
addpath('../Parameters')                                                            %neuron Parameters
addpath('../Network-Architecture')                                                  %neuron connectivity script
addpath(genpath('../../../../DynaSim'))                                             %DynaSim (Builds and runs ODEs given by previous scripts)
plot_all = 1;                                                                       %Temp  \TODO clean up the post processing for single channel.
dt = 0.1;                                                                           %Set time-step for DynaSim

%Generate C-Based files for speed (Mex)                                            
Mex_option = 1;                                                                     %\TODO get MEX working on single channel.
Mex_Prep;                                                                           %Generates MEX file structure. If not using MEX creates default structure

%Declare Parameter File
params_5_rate_based_onoff_fig4;                                                     %Parameter file for Onset Dominated Regieme in 2025 Modeling Paper.

% load Pre-Cortical Signal
load('../../PreCortical-Modeling/default_STRF_with_offset_200k.mat');
CompensateDT;                                                                       %Compensate if dt != 0.1 to align w/ experiments.

% create input spikes from STRFs
prepInputData;                                                                      %\TODO Comment and clean

% Run & Setup network architecture
setOptions;                                                                         %Set DynaSim Options \TODO Comment
[snn_out,s] = columnNetwork_paper_onoff(study_dir,varies,options,flag_raised_mex);  %\TODO Comment and clean

% post-process for performance and firing results
postProcessSims;                                                                    %\TODO Comment and clean


