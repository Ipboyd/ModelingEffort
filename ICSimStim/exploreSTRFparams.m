% exploring STRF parameters
addpath(genpath('strflab_v1.45'))

% get axes
[song1,fs1] = audioread('..\stimuli\200k_target1.wav');
[song1_spec,t,f]=STRFspectrogram(song1/rms(song1)*0.01,fs1);

% initial parameters
paramH.t0=7/1000; % s
paramH.BW=0.0045; % s temporal bandwith (sigma: exp width)
paramH.BTM=56; %56;  % Hz  temporal modulation (sine width)
paramH.phase=.49*pi;
paramG.BW=2000;  % Hz
paramG.BSM=5.00E-05; % 1/Hz=s
paramG.f0=4300;

% single strf?
% strf=STRFgen(paramH,paramG,f,t(2)-t(1));
% imagesc(t,f,strf.w1)

%
variedParam = 'BW';
range = 0.005:0.005:0.1;
for var = range
    paramH.BW = var;
    strf=STRFgen(paramH,paramG,f,t(2)-t(1));
    h = imagesc(strf.t,strf.f,strf.w1);
    colorbar;
    title(['paramH.' variedParam ' = ' num2str(var)])
    xlabel('time (s)')
    ylabel('freq (Hz')
    filename = ['STRF' filesep variedParam filesep variedParam num2str(var,'%0.4f') '.jpg'];
    saveas(h,filename);
end

% viewing results
% using SC package: https://github.com/ojwoodford/sc
% addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\sc-master');
% ims = imstream('paramH-BTM.mp4');
% imdisp(ims)

%% gabor functions
% addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils')
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils\MatPlotLib2.0 Colormaps')

% parameters:
paramH.BW= 0.05; %bandwidth
paramH.BTM= 3.8 ; %BTM, modulation
paramH.t0= 0.125; % t0, peak latency (s)
paramH.phase= 0.45*pi; % phase
    
dt = 0.001;
t = 0:dt:.250;

% parameter values to vary over
range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1];
colormap = inferno(length(range));
figure;
i = 1;
for var = range
    paramH.BW= var;
    h = exp(-0.5*((t-paramH.t0)/paramH.BW).^2).*cos(2*pi*paramH.BTM*(t-paramH.t0)+paramH.phase);
%     strf.exp=exp(-.5*((strf.t-paramH.t0)/paramH.BW).^2);
%     strf.cos=cos(2*pi*paramH.BTM*(strf.t-paramH.t0)+paramH.phase);
%     h=strf.exp.*strf.cos;
    plot(t,h,'color',colormap(i,:)); hold on;
    i = i+1;
end
legend(cellstr(num2str(range')));