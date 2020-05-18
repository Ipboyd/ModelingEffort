function t_spiketimes=InputGaussianSTRF_v2(specs,songloc,maskerloc,tuning,saveParam,mean_rate,stimGain,maxWeight)
% Inputs
%   specs - spectrogram representation of stimuli, with fields
%       .songs{2} for the two songs
%       .maskers{10} for the 10 masker trials
%       .t and .f for the time-frequency axes
%   songloc, maskerloc - a vector between 0 and 4
%   tuning - a structure, with fields
%       .type - 'bird' for gaussian tuning curves, or
%               'mouse' for mouse parameters
%       .sigma - tuning curve width
%       .H, .G - STRF parameters
%   saveParam  - a structure, with fields
%       .flag - save or not
%       .fileLoc - save file name
%   mean_rate - (?) mean firing rate?
%   stimGain - input stimulus gain
%   maskerlvl -
%
%
% modified by KFC
% 2019-08-07 added switch/case for two types of tuning curves
%            removed 1/2 scaling factor for colocated stimulus spectrograms
%            cleaned up input params
% V2:
% 2019-08-30 moved normalization to after spectrogram/tuning curve weighing
% 2019-08-31 replaced normalization with the gain parameter
% 2019-09-05 replaced song-shaped noise with white guassian noise
% 2019-09-10 recreate wgn for every trial & cleaned up code
% 2020-04-16 moved all .wav reads and spectrogram calculations to the main
%            code, to minimize redundancy; 
%            added SPECS input parameter;
%            made all default figures invisible to prevent focus stealing
% 2020-05-14 moved figure call to main code
%
% To do: spatial tuning curves can be moved to the main code too

% Plotting parameters
colormap = parula;
color1=colormap([1 18 36 54],:);
width=11.69;hwratio=.6;
x0=.05;y0=.1;
dx=.02;dy=.05;
lx=.13;ly=.1;
azimuth=[-90 0 45 90]; %stimuli locations
figuresize(width, width*hwratio, 'inches')
positionVector = [x0+dx+lx y0+dy+ly 5*lx+4*dx ly];
subplot('Position',positionVector)
hold on
annotation('textbox',[.375 .33 .1 .1],...
    'string',{['\sigma = ' num2str(tuning.sigma) ' deg'],['gain = ' num2str(stimGain)]},...
    'FitBoxToText','on',...
    'LineStyle','none')

% other parameters
if saveParam.flag, savedir=[saveParam.fileLoc]; mkdir(savedir); end

% Define spatial tuning curves & plot
sigma = tuning.sigma;
switch tuning.type
    case 'bird'
        x=-108:108;
        tuningcurve=zeros(4,length(x));
        tuningcurve(1,:)=gaussmf(x,[sigma,-90]);
        tuningcurve(2,:)=gaussmf(x,[sigma,0]);
        tuningcurve(3,:)=gaussmf(x,[sigma,45]);
        tuningcurve(4,:)=gaussmf(x,[sigma,90]);
        neuronNames = {'-90d deg','0 deg','45 deg','90 deg'};
    case 'mouse'
        x=-108:108;
        tuningcurve=zeros(4,length(x));
        tuningcurve(1,:)= sigmf(-x,[0.016 -22.5])-0.05; %flipped sigmodial
        tuningcurve(2,:)=gaussmf(x,[sigma,0]); %guassian
        tuningcurve(3,:)= 1- gaussmf(x,[sigma,0]); %U shaped gaussian
        tuningcurve(4,:)= sigmf(x,[0.016 -22.5])-0.05; %sigmodial
        neuronNames = {'left sigmoid','gaussian','U','right sigmoid'};
end

for i=1:4
    plot(x,tuningcurve(i,:),'linewidth',2.5,'color',color1(i,:))
end
xlim([min(x) max(x)]);ylim([0 1.05])
set(gca,'xtick',[-90 0 45 90],'XTickLabel',{'-90 deg', '0 deg', '45 deg', '90 deg'},'YColor','w')
set(gca,'ytick',[0 0.50 1.0],'YTickLabel',{'0', '0.50', '1.0'},'YColor','b')

% ---- initialize stimuli spectrogram ----
masker_spec = specs.maskers{1};
t = specs.t;
f = specs.f;

% plot STRF
strf = tuning.strf;

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

%%
t_spiketimes={};
avgSpkRate=zeros(1,4);disc=zeros(1,4);
for songn=1:2
    %convert sound pressure waveform to spectrogram representation
%     songs(:,songn)=songs{songn}(1:n_length);
%     [song_spec,~,~]=STRFspectrogram(songs{songn},fs);
    song_spec = specs.songs{songn};

    %% plot mixture process (of song1) for visualization
    stim_spec=zeros(4,specs.dims(1),specs.dims(2));
    if maskerloc
        stim_spec(maskerloc,:,:)=masker_spec;
    end

    if songloc
        % when masker and song are colocated
        if maskerloc==songloc
            stim_spec(songloc,:,:)=(masker_spec + song_spec)/2;
        else
            stim_spec(songloc,:,:)=song_spec;
        end
    end
    if songn==1
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
    mixedspec=zeros(size(stim_spec));
    weight=zeros(4,4);
    for i=1:4  % summing of each channel, i.e. neuron type 1-4
        for trial = 1:10         % for each trial, define a new random WGN masker
%             masker = wgn(1,n_length,1);
            masker_spec = specs.maskers{trial};

            %% weight at each stimulus location
            totalWeight = 0;
            if songloc
                weight(i,songloc) = tuningcurve(i,x==azimuth(songloc));
                totalWeight = totalWeight + weight(i,songloc);
                mixedspec(i,:,:) = squeeze(mixedspec(i,:,:)) + weight(i,songloc)*song_spec;
            end
            if maskerloc
                weight(i,maskerloc) = tuningcurve(i,x==azimuth(maskerloc));
                totalWeight = totalWeight + weight(i,maskerloc);
                mixedspec(i,:,:) = squeeze(mixedspec(i,:,:)) + weight(i,maskerloc)*masker_spec;
            end

            % scale mixed spectrogram; cap total weight to maxWeight
            if totalWeight <= maxWeight
              mixedspec(i,:,:) = mixedspec(i,:,:)*stimGain;
            else
              mixedspec(i,:,:) = mixedspec(i,:,:)/totalWeight*maxWeight*stimGain;
            end
            %mixedspec(i,:,:) = mixedspec(i,:,:).*stimGain;

            if i>1
                subplotloc=i+1;
            else
                subplotloc=i;
            end

            currspec=squeeze(mixedspec(i,:,:)); % currentspectrograms

            %% plot mixed spectrograms (of song1)- 3rd row of graphs
            if songn==1 && trial==1
                positionVector = [x0+subplotloc*(dx+lx) y0+2*(dy+ly) lx ly];
                subplot('Position',positionVector)
                imagesc(t,f,currspec',[0 80]);colormap('parula');
                xlim([0 max(t)])
                set(gca,'YDir','normal','xtick',[0 1],'ytick',[])
            end

            %% convolve STRF with spectrogram
            [spkcnt,rate,tempspk]=STRFconvolve(strf,currspec,mean_rate,1,songn);
            avgSpkRate(i)=spkcnt/max(t);
            fr{trial,i+4*(songn-1)} = rate;
            t_spiketimes{trial,i+4*(songn-1)} = tempspk; %sec
        end

        %% plot FR (of song1)
        %2nd row of plots- spectograph
        if songn==1
            positionVector = [x0+subplotloc*(dx+lx) y0+3*(dy+ly) lx ly];
            subplot('Position',positionVector)
            plot(t,rate);xlim([0 max(t)])
        end
        % raster plot- first row of graphs
        positionVector = [x0+subplotloc*(dx+lx) y0+4*(dy+ly) lx ly];
        subplot('Position',positionVector);hold on
        %The below for loop codes for the first row of plots with the rasters.
        for trial=1:10
            raster(t_spiketimes{trial,i+4*(songn-1)},trial+10*(songn-1)) %need to change tempspk to change the raster
        end
        plot([0 2000],[10 10],'k')
        ylim([0 20])
        xlim([0 max(t)*1000])
        %Below section gives the whole row of top labels
        if songn==2
            distMat = calcvr([t_spiketimes(:,i) t_spiketimes(:,i+4)], 10); % using ms as units, same as ts
            [disc(i), E, correctArray] = calcpc(distMat, 10, 2, 1,[], 'new');
            firingRate = round(sum(cellfun(@length,t_spiketimes(:,i+4)))/10);
            title({neuronNames{i},['disc = ', num2str(disc(i))],['FR = ',num2str(firingRate)]})
        end

        fclose all;
    end

    if saveParam.flag
        saveas(gca,[savedir '/s' num2str(songloc) 'm' num2str(maskerloc) '.tiff'])
        save([savedir '/s' num2str(songloc) 'm' num2str(maskerloc)],'t_spiketimes','songloc','maskerloc',...
            'sigma','mean_rate','disc','avgSpkRate','fr')
    end
end
clf