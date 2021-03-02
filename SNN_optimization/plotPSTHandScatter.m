figure('position',[200 200 1200 500]);
xstart = 0.1;
ystart = 0.1;
height = 0.2;
width = 0.5;
i = 1;
for configNum = configsToPlot
    subplot('position',[0.05 0.9-height*i width height]);
    startIdx = configNum*trialLen-trialLen+1;
    endIdx = startIdx+trialLen-1;
    currentSong = input_norm(startIdx:endIdx,:);
    max(currentSong)
    out_ideal = predict(ideal_net,currentSong);
    out_learned = predict(net,currentSong);
    out_SNN = output_norm(startIdx:endIdx);
    
    plot(currentSong(:,i),'linewidth',2); hold on;
    plot(out_ideal);
    plot(out_learned);
    plot(out_SNN);
    xticks([])
    
    if i==1
        leg1 = legend('training  input','output with ideal weights','output with learned weights','SNN output');
        leg1.Location = 'northeast';
    end
    subplot('position',[0.575 0.1 0.4 0.8])
    scatter(currentSong(:,i),out_SNN); hold on;
    leg2 = legend('Input vs SNN  output');
    i = i+1;
end

suptitle(['training set ' num2str(trainingSetNum) '; ' plotIdentifier])
saveas(gcf,['PSTH training set ' num2str(trainingSetNum) ' - ' plotIdentifier '.tif'])