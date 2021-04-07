load('data_recordings\PSTH_IC_clean.mat')

nonEmptyChannels = 1:32;
nonEmptyChannels(cellfun(@isempty,Spks_clean)') = [];

%% find max lag
for loc = 1:4
    for chan = nonEmptyChannels
        [r,lags] = xcorr(psth1clean{4},psths1{loc,chan});
        maxR(loc,chan) = max(r);
        maxLag(loc,chan) = lags(r==max(r));

        [r,lags] = xcorr(psth2clean{4},psths2{loc,chan});
        maxR2(loc,chan) = max(r);
        maxLag2(loc,chan) = lags(r==max(r));
    end
    round([(1:32)',maxR(loc,:)', maxLag(loc,:)', maxR2(loc,:)', maxLag2(loc,:)'])
end

round([(1:32)',maxLag'])
round([(1:32)',maxLag2'])
%% visualize psths between clean and raw data
interestedChannels = 27;
for chan = interestedChannels
    figure;
    s1delay = maxLag(1,chan);
    s2delay = maxLag2(1,chan);
    %plot IC PSTHs
    for loc = 1
        subplot(2,1,1)
        plot(psth1clean{loc}); hold on;
        subplot(2,1,2)
        plot(psth2clean{loc}); hold on;
    end
    %plot data PSTHs
    for loc = [1]
        subplot(2,1,1)
        plot([zeros(1,s1delay),psths1{loc,chan}]); hold on;
        subplot(2,1,2)
        plot([zeros(1,s2delay),psths2{loc,chan}]); hold on;
    end
    legend('IC, 90deg','Data 0deg','Data -90deg')
    sgtitle(sprintf('channel %i',chan'))
end
%% plot performance
best_tau = 5;
channel_to_model = 2;
cleanPerf = clean_performance{channel_to_model}(best_tau,:)
mixedPerf = squeeze(masked_performance{channel_to_model}(best_tau,:,:))

figure;
[X,Y] = meshgrid(1:4,1:4);
subplot('position',[0.1 0.75 0.8 0.15])
imagesc(cleanPerf)
text((1:4)-0.25,ones(1,4),num2str(cleanPerf'))
xticklabels([])
yticklabels([])
caxis([50 100])
subplot('position',[0.1 0.1 0.8 0.6])
imagesc(mixedPerf)
xticks([1,2,3,4])
xticklabels({'90','45','0','-90'})
yticks([1,2,3,4])
yticklabels({'-90','0','45','90'})
ylabel('masker location')
xlabel('song location')
text(X(:)-0.25,Y(:),num2str(mixedPerf(:)))
caxis([0 100])