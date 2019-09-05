function t_spiketimes=InputGaussianSTRF(songloc,maskerloc,tuning,saveParam,mean_rate,stimGain)
% Inputs
%   songloc, maskerloc - a vector between 0 and 4
%   tuning - a structure, with fields
%       .type - 'bird' for gaussian tuning curves, or 
%               'mouse' for mouse parameters
%       .sigma - tuning curve width
%       .H, .G - STRF parameters
%   saveParam  - a structure, with fields
%       .flag - save or not
%       .fileLoc - save file name
%   mean_rate - ?
%   stimGain - input stimulus gain
%
%
% modified by KFC
% 2019-08-07 added switch/case for two types of tuning curves
%            removed 1/2 scaling factor for colocated stimulus spectrograms
%            cleaned up input params
% 2019-08-30 moved normalization to after spectrogram/tuning curve weighing

% Plotting parameters
colormap = parula;
color1=colormap([1 18 36 54],:);
width=11.69;hwratio=.6;
x0=.05;y0=.1;
dx=.02;dy=.05;
lx=.13;ly=.1;
azimuth=[-90 0 45 90];
h=figure;
figuresize(width, width*hwratio, 'inches')
positionVector = [x0+dx+lx y0+dy+ly 5*lx+4*dx ly];
subplot('Position',positionVector)
hold on
annotation('textbox',[.375 .33 .1 .1],...
    'string',{['\sigma = ' num2str(tuning.sigma) ' deg'],['gain = ' num2str(stimGain)]},...
    'FitBoxToText','on',...
    'LineStyle','none')

% Define spatial tuning curves
sigma = tuning.sigma;
switch tuning.type
    case 'bird'
        x=-108:108;
        tuningcurve=zeros(4,length(x));
        tuningcurve(1,:)=gaussmf(x,[sigma,-90]);
        tuningcurve(2,:)=gaussmf(x,[sigma,0]);
        tuningcurve(3,:)=gaussmf(x,[sigma,45]);
        tuningcurve(4,:)=gaussmf(x,[sigma,90]);
    case 'mouse'
        x=-108:108;
        tuningcurve=zeros(4,length(x));
        tuningcurve(1,:)= sigmf(-x,[0.016 -22.5])-0.05; %flipped sigmodial
        tuningcurve(2,:)=gaussmf(x,[sigma,0]); %guassian
        tuningcurve(3,:)= 1- gaussmf(x,[sigma,0]); %U shaped gaussian
        tuningcurve(4,:)= sigmf(x,[0.016 -22.5])-0.05; %sigmodial
end


%% Load songs
load('stimuli.mat','stimuli')
fs=44100;  % takes song 2
n_length=length(stimuli{2});%t_end=length(song2/fs);
songs=zeros(n_length,2);
masker=stimuli{3}(1:n_length); %creates masker (stored in stimuli.mat{3}) of length song2

%% masker spectrogram (is fixed)
[masker_spec,t,f]=STRFspectrogram(masker,fs);

%% Make STRF
strf=STRFgen(tuning.H,tuning.G,f,t(2)-t(1));

%% Save the figure and figure variables in the STRF folder
if saveParam.flag, savedir=[tuning.type '\' saveParam.fileLoc]; mkdir(savedir); end

t_spiketimes={};
spkrate=zeros(1,4);disc=zeros(1,4);
%% 
for songn=1:2
    %convert sound pressure waveform to spectrogram representation
    stim_spec=zeros(4,length(t),length(f));
    if maskerloc
        stim_spec(maskerloc,:,:)=masker_spec;
    end
    songs(:,songn)=stimuli{songn}(1:n_length);
    [song_spec,~,~]=STRFspectrogram(songs(:,songn),fs);
    if songloc
        % when masker and song are colocated
        if maskerloc==songloc
            stim_spec(songloc,:,:)=(masker_spec + song_spec)/2;
        else
            stim_spec(songloc,:,:)=song_spec;
        end
    end
    stim_spec = stim_spec*stimGain;
    %% plot mixture process (of song1) for visualization
    if songn==1

        % plot Gaussian curves
       
        for i=1:4
            plot(x,tuningcurve(i,:),'linewidth',2.5,'color',color1(i,:))
        end
        xlim([min(x) max(x)]);ylim([0 1.05])
        set(gca,'xtick',[-90 0 45 90],'XTickLabel',{'-90 deg', '0 deg', '45 deg', '90 deg'},'YColor','w')
        set(gca,'ytick',[0 0.50 1.0],'YTickLabel',{'0', '0.50', '1.0'},'YColor','b')
       
        % plot STRF
        positionVector = [x0 y0+2*(dy+ly) lx/2 ly];
        subplot('Position',positionVector)
        set(gca,'ytick',[4000 8000],'yTickLabel',{'4','8'})
        imagesc(strf.t, strf.f, strf.w1); axis tight;%colorbar
        axis xy;
        v_axis = axis;
        v_axis(1) = min(strf.t); v_axis(2) = max(strf.t);
        v_axis(3) = min(strf.f); v_axis(4) = max(strf.f);
        axis(v_axis);
        xlabel('t (sec)')
        title('STRF')
        
        % plot spectrograms for song1- bottom row of graphs
        for i=1:4
            %the below if statement creates the space in between the first graph and the other 3
            if i>1
                subplotloc=i+1;
            else
                subplotloc=i;
            end
            
            positionVector = [x0+subplotloc*(dx+lx) y0 lx ly];
            subplot('Position',positionVector)
            imagesc(t,f,squeeze(stim_spec(i,:,:))',[0 80]);colormap('parula');
            xlim([0 max(t)])
            set(gca,'YDir','normal','xtick',[0 1],'ytick',[])
        end
    end
    %% mix spectrograms using Gaussian weights
    % sum spectrograms
    mixedspec=zeros(size(stim_spec));
    weight=zeros(4,4);
    for i=1:4  % summing of each channel
        %% weight at each stimulus location
        for j=1:4 
            weightindex=find(x==azimuth(j));
            weight(i,j)=tuningcurve(i,weightindex);
            mixedspec(i,:,:)=weight(i,j)*stim_spec(j,:,:)+mixedspec(i,:,:);
        end

        if i>1
            subplotloc=i+1;
        else
            subplotloc=i;
        end
       
        
%         sat_old=max([max(max(stim_spec(1,:,:))) max(max(stim_spec(2,:,:))) max(max(stim_spec(3,:,:))) max(max(stim_spec(4,:,:)))]);
% 
%         normweight=any(max(max(stim_spec(1,:,:))))*weight(i,1)+any(max(max(stim_spec(2,:,:))))*weight(i,2)+any(max(max(stim_spec(3,:,:))))...
%             *weight(i,3)+any(max(max(stim_spec(4,:,:))))*weight(i,4);
%         mixedspec(i,:,:)=mixedspec(i,:,:)/normweight;

        currspec=squeeze(mixedspec(i,:,:)); % currentspectrograms
        %% plot mixed spectrograms (of song1)- 3rd row of graphs
        if songn==1
            positionVector = [x0+subplotloc*(dx+lx) y0+2*(dy+ly) lx ly];
            subplot('Position',positionVector)
            imagesc(t,f,currspec',[0 80]);colormap('parula');%colorbar
            xlim([0 max(t)])
            set(gca,'YDir','normal','xtick',[0 1],'ytick',[])
        end
        %% convolve STRF with spectrogram
        [spkcnt,rate,tempspk]=STRFconvolve(strf,currspec,mean_rate,10,songn);
        spkrate(i)=spkcnt/max(t);
        t_spiketimes=[t_spiketimes tempspk]; %sec
        %% plot FR (of song1)
        %2nd row of plots- spectograph
        if songn==1
            positionVector = [x0+subplotloc*(dx+lx) y0+3*(dy+ly) lx ly];
            subplot('Position',positionVector)
            plot(t,rate);xlim([0 max(t)])
        end
        %% raster plot- first row of graphs
        positionVector = [x0+subplotloc*(dx+lx) y0+4*(dy+ly) lx ly];
        subplot('Position',positionVector);hold on
        %The below for loop codes for the first row of plots with the rasters.
        for ir=1:10 
            raster(tempspk{ir,1},ir+10*(songn-1)) %need to change tempspk to change the raster
        end
        plot([0 2000],[10 10],'k')
        ylim([0 20])
        xlim([0 max(t)*1000])
        %Below section gives the whole row of top labels
        if songn==2
            distMat = calcvr([t_spiketimes(:,i) t_spiketimes(:,i+4)], 10); % using ms as units, same as ts
            [disc(i), E, correctArray] = calcpc(distMat, 10, 2, 1,[], 'new');
            title({['disc = ', num2str(disc(i))],['FR = ',num2str(spkrate(i))]})
        end
    end   

    if saveParam.flag
        saveas(gca,[savedir '/s' num2str(songloc) 'm' num2str(maskerloc) '.tiff'])
        paramG = tuning.G;
        paramH = tuning.H;
        save([savedir '/s' num2str(songloc) 'm' num2str(maskerloc)],'t_spiketimes','songloc','maskerloc',...
            'sigma','paramG','paramH','mean_rate','disc','spkrate')
    end
end