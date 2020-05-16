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
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils')
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils\MatPlotLib2.0 Colormaps')
color1 = brewermap(3,'set1');

dt = 0.001;
t = 0:dt:.250;

% parameter values to vary over
range = cell(4);
range{1} = 0.05:.01:0.15; %t0
range{2} = [2:0.5:10,56]; %BTM
range{3} = [0.4:0.01:0.5]; %phase
range{4} = 0.0045:0.0005:0.01; %BW
variedParam = {'t0','BTM','phase','BW'};
figure;
for j = 1:length(range)
    subplot(2,2,j);
    colormap = inferno(length(range{j}));
    i = 1;
    % default parameters:
    paramH.BW= 0.03; %bandwidth
    paramH.BTM= 3.8 ; %BTM, modulation
    paramH.t0= 0.1; % t0, peak latency (s)
    paramH.phase= 0.5; % phase

    for var = range{j}
        eval(['paramH.' variedParam{j} ' = var;']);
        hGauss = exp(-0.5*((t-paramH.t0)/paramH.BW).^2);
        hcos = cos(2*pi*paramH.BTM*(t-paramH.t0)+paramH.phase*pi);
        h = hGauss.*hcos;
        plot(t,h,'color',colormap(i,:)); hold on;
        i = i+1;
    end
    hleg = legend(cellstr(num2str((range{j})')));
    title(variedParam{j})
end

% plot(t,hGauss,'color',color1(1,:))
% plot(paramH.t0,1,'o','color',color1(1,:),'markerfacecolor',color1(1,:))
% text(paramH.t0,1-0.1,['(' num2str(paramH.t0,'%0.3f') ',1)'])
% hleg = legend(cellstr(num2str((range)')));

%% IC response simulations
close all;
%!!!!!!!!! run the first chunk of code from inputGaussianSTRF_main !!!!!!!%

% % % default parameters:
paramH.BW= 0.05; %bandwidth
paramH.BTM= 3.8 ; %BTM, modulation
paramH.t0= 0.1; % t0, peak latency (s)
paramH.phase= 0.5*pi; % phase
strfGain = 0.1;

range = [0.48 0.49 0.495 0.497 0.498 0.4985 0.499 0.4995 0.5]*pi; %phase
set(0, 'DefaultFigureVisible', 'off')
figure;
for var = range
    paramH.phase = var;
    strf=STRFgen(paramH,paramG,f,t(2)-t(1));
    strf.w1 = strf.w1*strfGain;
    
    saveName = sprintf('BW_%0.3f BTM_3.8 t0_0.1\\s%d_STRFgain%0.2f_phase%0.4f_%s',paramH.BW,sigma,strfGain,paramH.phase/pi,datestr(now,'YYYYmmdd-HHMMSS'));
    saveParam.flag = 1;
    saveParam.fileLoc = [dataSaveLoc filesep tuning filesep saveName];
    if ~exist(saveParam.fileLoc,'dir'), mkdir(saveParam.fileLoc); end

    tuningParam.strf = strf;
    tuningParam.type = tuning;
    tuningParam.sigma = sigma;
    t_spiketimes=InputGaussianSTRF_v2(specs,1,0,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
    t_spiketimes=InputGaussianSTRF_v2(specs,0,1,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
    t_spiketimes=InputGaussianSTRF_v2(specs,1,4,tuningParam,saveParam,mean_rate,stimGain,maxWeight);
end
set(0, 'DefaultFigureVisible', 'on')

%% plot the results from the above cell
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils\MatPlotLib2.0 Colormaps')
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\plotting utils')
addpath('C:\Users\Kenny\Dropbox\Sen Lab\m-toolboxes\IoSR')

% neuron firing rates to visualize
neuronNum = 4; %[LSigmoid, gauss, U, RSigmoid]

% grab data
temp = strsplit(saveName,'\')
datafiles = dir([dataSaveLoc filesep tuning filesep temp{1}])

ind{1} = 3:length(datafiles);
% ind{2} = 16:length(datafiles);
for k = 1:length(ind)
    clear fr label perf
    j = 1;
    figure;
    for i = ind{k}
        data = load([datafiles(i).folder filesep datafiles(i).name filesep 's1m4.mat']);
        fr(j,:) = data.fr{neuronNum}/data.avgSpkRate(neuronNum); %grab firing rate
        label{j} = datafiles(i).name(23:28); %grab label
        perf(j) = data.disc(neuronNum);
        avgfr(j) = data.avgSpkRate(neuronNum);
        j = j+1;
    end
    fr(isinf(fr)) = 0;
    colormap = brewermap(j-1,'dark2');
    h = iosr.figures.multiwaveplot(fr,'reverseY',1);
    for i = 1:j-1
        set(h(i),'color',colormap(i,:)); 
        text(50,i-0.7,sprintf('perf=%0.2f, FR=%0.2f, phase=%s\\pi', perf(i),avgfr(i),label{i}),'fontsize',12)
    end
    yticks([])
    xticks([])
    set(gcf,'Position',[100 0 500 1000])
end