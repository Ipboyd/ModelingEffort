function [t_spiketimes_on,t_spiketimes_off]=InputGaussianSTRF_v4...
    (specs,songloc,maskerloc,tuning,saveParam,mean_rate,stimGain,maxWeight,paramSpk)
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
% 2021-01-07 simplified code for spectrogram mixing
%
% V3:
% 2021-08-03 reversed x-axis so that +45˚ and +90˚ is ccw to match mouse
%            exps, replaced VR-based perf with SPIKE-distance-based perf
%
% V4: 
% 2021-12-06 now code for offset (frate<0 in STRFconvolve)
% 2021-12-10 broadened tuning curves to match Ono and Oliver data
% 
% To do: spatial tuning curves can be moved to the main code too

% Plotting parameters
colormap = parula;
color1=colormap([1 18 36 54],:);
width=11.69;hwratio=.6;
x0=.05;y0=.1;
dx=.02;dy=.05;
lx=.13;ly=.1;
azimuth = fliplr([-90 0 45 90]); %stimuli locations (flipped to match mouse data)
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
x = -108:108;
tuningcurve=zeros(4,length(x));

% tuningcurve(1,:) = sigmf(x,[0.07 50])/sigmf(90,[0.07 50]); % = 1 at x = +90
% tuningcurve(2,:) = 1 - gaussmf(x,[sigma 0]);
% tuningcurve(3,:) = gaussmf(x,[sigma 0]);
% tuningcurve(4,:) = sigmf(x,[-0.07 -50])/sigmf(-90,[-0.07 -50]); % at -90°, sigmoid == 1
neuronNames = fliplr({'ipsi sigmoid','gaussian','U','contra sigmoid'});

tuningcurve(1,:) = sigmf(x,[0.015 60])/sigmf(90,[0.015 60]); % = 1 at x = +90
tuningcurve(2,:) = gaussmf(x,[sigma 45])*0.8 + 0.2;% 1.2 - gaussmf(x,[sigma 0])*0.8;
tuningcurve(3,:) = gaussmf(x,[sigma 0])*0.8 + 0.2;
tuningcurve(4,:) = sigmf(x,[-0.015 -60])/sigmf(-90,[-0.015 -60]); % at -90°, sigmoid == 1

for i=1:4
    plot(x,tuningcurve(i,:),'linewidth',2.5,'color',color1(i,:))
end
xlim([min(x) max(x)]);ylim([0 1.05]); set(gca,'xdir','reverse');
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
t_spiketimes_on={}; t_spiketimes_off={};
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
            if i > 3
                subplotloc = i+1;
            else
                subplotloc = i;
            end

            positionVector = [x0+subplotloc*(dx+lx) y0 lx ly];
            subplot('Position',positionVector)
            imagesc(t,f,squeeze(stim_spec(i,:,:))',[0 80]);colormap('parula');
            xlim([0 max(t)])
            set(gca,'YDir','normal','xtick',[0 1],'ytick',[])
        end
    end


    %% mix spectrograms using Gaussian weights
    for i=1:4  
        % initialize weights
        maskerWeight = 0;
        songWeight = 0;
        % for each tuning curve, i.e. neuron type 1-4, 
        % 1. weight stimulus at each location by tuning curve amplitude
        % 2. sum weighted stimuli
        for trial = 1:10         % for each trial, define a new random WGN masker
%             masker = wgn(1,n_length,1);
            masker_spec = specs.maskers{trial};
            
            % for each location along the azimuth, weight stimuli spectrogram
            % 2020-01-05 separate colocated configs

            %%%%%%%%%%%%%%
            if maskerloc, maskerWeight = tuningcurve(i,x==azimuth(maskerloc)); end
            if songloc, songWeight = tuningcurve(i,x==azimuth(songloc)); end
            totalWeight = maskerWeight + songWeight;

            mixed_spec = masker_spec*maskerWeight + song_spec*songWeight;

            % cap total weight to maxWeight, then scale
            if totalWeight > maxWeight
                mixed_spec = mixed_spec/totalWeight*maxWeight;
            end
            mixed_spec = mixed_spec * stimGain;
            %%%%%%%%%%%%%%%

            if i > 3
                subplotloc=i+1;
            else
                subplotloc=i;
            end

%             currspec=squeeze(mixedspec(i,:,:)); % currentspectrograms
            
            currspec = mixed_spec;
            %% plot mixed spectrograms (of song1)- 3rd row of graphs
            if songn==1 && trial==1
                positionVector = [x0+subplotloc*(dx+lx) y0+2*(dy+ly) lx ly];
                subplot('Position',positionVector)
                imagesc(t,f,currspec',[0 80]);colormap('parula');
                xlim([0 max(t)])
                set(gca,'YDir','normal','xtick',[0 1],'ytick',[])
            end

            %% convolve STRF with spectrogram for onset and offset firing
            [spkcnt_on,spkcnt_off,frate_on,frate_off,temp_on,temp_off] = ...
                STRFconvolve_V2(strf,currspec,mean_rate,1,songn,paramSpk.t_ref,paramSpk.t_ref_rel,paramSpk.offsetFrac,paramSpk.rec);
            
            avgSpkRate_on(i)=spkcnt_on/max(t);
            fr_on{trial,i+4*(songn-1)} = frate_on;
            t_spiketimes_on{trial,i+4*(songn-1)} = temp_on; %sec
            
            avgSpkRate_off(i)=spkcnt_off/max(t);
            fr_off{trial,i+4*(songn-1)} = frate_off;
            t_spiketimes_off{trial,i+4*(songn-1)} = temp_off; %sec
            
        end

        %% plot FR (of song1)
        %2nd row of plots- spectograph
        if songn==1
            positionVector = [x0+subplotloc*(dx+lx) y0+3*(dy+ly) lx ly];
            subplot('Position',positionVector)
            
            % plot firing rate or PSTH
            % plot(t,frate_on); xlim([0 max(t)]);
            psth(vertcat(t_spiketimes_on{:,i+4*(songn-1)}),t);
            
        end
        % raster plot- first row of graphs
        positionVector = [x0+subplotloc*(dx+lx) y0+4*(dy+ly) lx ly];
        subplot('Position',positionVector);hold on
        %The below for loop codes for the first row of plots with the rasters.
        for trial=1:10
            raster(t_spiketimes_on{trial,i+4*(songn-1)},trial+10*(songn-1)) %need to change tempspk to change the raster
        end
        plot([0 max(t)*1000],[10 10],'k')
        ylim([0 20])
        xlim([0 max(t)*1000])
        
        %Below section gives the whole row of top labels
        if songn==2
            input = cellfun(@(x) x',[t_spiketimes_on(:,i)' t_spiketimes_on(:,i+4)'],'uniformoutput',false); % should be 1x20 cell array of row vectors
            STS = SpikeTrainSet(input,250,max(t)*1000);
            distMat = STS.SPIKEdistanceMatrix(250,max(t)*1000);
            disc(i) = calcpc(distMat, 10, 2, 1,[], 'new');
            firingRate = round(mean(cellfun(@length,t_spiketimes_on(:,i+4)))/(max(t)-0.250));
            title({neuronNames{i},['disc = ', num2str(disc(i))],['FR = ',num2str(firingRate)]})
        end

        fclose all;
    end

    if saveParam.flag
        saveas(gca,[savedir '/s' num2str(songloc) 'm' num2str(maskerloc) '.tiff'])
        save([savedir '/s' num2str(songloc) 'm' num2str(maskerloc)],'t_spiketimes_on','t_spiketimes_off','songloc','maskerloc',...
            'sigma','mean_rate','disc','avgSpkRate_on','avgSpkRate_off','fr_on','fr_off')
    end
end
clf