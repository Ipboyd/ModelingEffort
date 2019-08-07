%Script that runs all the functions and creates the desired output calls
%the function 'InputGaussianSTRF'
% Script might not run because of direction of backslashes.
% Make sure your matlab version has the Fuzzy Logic Toolbox downloaded
% already

%To run the file, input all these files into the same folder as this script
clear all;clc;close all
path = 'Z:\eng_research_hrc_binauralhearinglab\Model-Junzi_files_backup-remove_when_copied';
addpath([path 'scripts'])
addpath([path 'STRFs'])
addpath([path 'spikeGen'])
addpath([path 'Ross Code/software/genlib/'])
addpath([path 'Reconstruction/JD StimRecon'])
addpath(genpath('strflab_v1.45'))
colormap = parula;
% setfiguredefaults('pb')

% tuning curve parameters
sigma = 20; %60 for bird but 38 for mouse

%% STRF parameters

paramH.t0=7/1000; % s
paramH.BW=0.0045; % s temporal bandwith (sigma: exp width)
paramH.BTM=56;  % Hz  temporal modulation (sine width)
paramH.phase=.49*pi;

paramG.BW=2000;  % Hz
paramG.BSM=5.00E-05; % 1/Hz=s
paramG.f0=4300;

%%
% Calls the function InputGaussianSTRF to create files in the STRF folder with the generated figures 
mean_rate=.1;
datetime=datestr(now,'HHMMSS');

songLocs = 1:4;
maskerLocs = 1:4;
for songloc = songLocs
    close all
    maskerloc=0;
    t_spiketimes=InputGaussianSTRF(datetime,songloc,maskerloc,sigma,paramH,paramG,mean_rate);
    t_spiketimes=InputGaussianSTRF(datetime,maskerloc,songloc,sigma,paramH,paramG,mean_rate);
    for maskerloc = maskerLocs
        t_spiketimes=InputGaussianSTRF(datetime,songloc,maskerloc,sigma,paramH,paramG,mean_rate);
    end
end