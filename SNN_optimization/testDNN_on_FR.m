%% Prep data
trainingSetNum = 27;
trainingFileName = ['training_set_' num2str(trainingSetNum) '_noU.mat'];
load(trainingFileName,'input_training','output_training','netcons','trialLen')

input_norm = input_training/max(output_training(:));
output_norm = output_training/max(output_training(:));

input_norm(:,3) =  0; %zero out 3rd channel

% ====================== important parameters ========================
maxEpochs = 3;
lambda = 0.0002;
% configs to train
% trainConfigIdx = [1:4,5,10,15,20,6,12,18,24]; %single channel only
% trainConfigIdx = [1:4,5,10,15,20]; %target only
% trainConfigIdx = [5,10,7,11];
trainConfigIdx = 1:24;

IRWeightMask = ones(4)-eye(4);
IRWeightMask(3,:) = 0;
IRWeightMask(:,3) = 0;
% ======================================================================

songOnly = [];
trialLen = 32000-17;
for sampleStart = [1,size(input_training,1)/2+1]
    for i = trainConfigIdx
        songOnlyStart = i*trialLen-trialLen+sampleStart;
        songOnlyEnd = songOnlyStart+trialLen-1;
        songIdx{i} = {songOnlyStart,songOnlyEnd};
        songOnly = [songOnly,songOnlyStart:songOnlyEnd];
    end
end
X = input_norm(songOnly,:);
Y = output_norm(songOnly);

% % remove null samples
% nullSamples = (output_training < 0.001);
% X = input_norm(~nullSamples,:);
% Y = output_norm(~nullSamples);
trainBatch = 0.9;

m = size(X,1);
m_train = randsample(m,round(m*trainBatch));
Xtrain = X(m_train,:);
Ytrain = Y(m_train,:);
Xtest = X;
Xtest(m_train,:) = [];
Ytest = Y;
Ytest(m_train,:) = [];

%% Create Layers
% define main path
mu = 0.5;
a = mu - 0.005;
b = mu + 0.005;

mainPath = [
    featureInputLayer(4,'Name','input')
    
    additionLayer(2,'Name','add_In_Inh')
    reluLayer('Name','R_out')
    
%     fullyConnectedLayer(1,'Name','C','Weights',a + (b-a).*rand(1,4))
%     FCpositiveWeights(4,1,'C',a + (b-a).*rand(1,4))
    FCpositiveWeights(4,1,'C',ones(1,4))
    
    reluLayer('Name','C_out')
    regressionLayer('Name','output')
    ]

inhPath = [    
%     fullyConnectedLayer(4,'Name','R','Weights',a + (b-a).*rand(4))
    myReluExp('FI_approx')
    FCInhibitoryInput(4,4,'I->R',a + (b-a).*rand(4),IRWeightMask)
%     dropoutLayer('Name','dropout');
%     FCInhibitoryInput(4,4,'I->R',netcons.xrNetcon)
%     FCInhibitoryInput(4,4,'I->R',zeros(4))
    ]

lgraph = layerGraph(mainPath);
plot(lgraph);

% Add inh path to main path
lgraph = addLayers(lgraph,inhPath)
lgraph = connectLayers(lgraph,'input','FI_approx')
lgraph = connectLayers(lgraph,'I->R','add_In_Inh/in2')

plot(lgraph)

% freeze specified layers - doesn't actually work?!
% setLearnRateFactor(lgraph.Layers(7),'Weights',0)
% getLearnRateFactor(lgraph.Layers(7),'Weights') %???
% lgraph.Layers(8).Weights;
%% Analyze network
% analyzeNetwork(lgraph);

%% Train network
miniBatchSize = 512;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'L2Regularization',lambda, ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'ValidationData',{Xtest,Ytest}, ...
    'ValidationFrequency',300, ...
    'Verbose',0, ...
    'ExecutionEnvironment', 'auto',...
    'Plots','training-progress');

net = trainNetwork(Xtrain,Ytrain,lgraph,options);


%% Check network weights
net.Layers

IR_LayerIdx = 8  %choose appropriate index to get the desired layer
net.Layers(IR_LayerIdx).Weights

%% Train Network again with rectified weights

mu = 0.01;
a = mu - 0.005;
b = mu + 0.005;
for i = 1:1
    continue;
    subplot(1,3,i);
    imagesc(net.Layers(IR_LayerIdx).Weights); colorbar;
    
    % recfity weights
    temp_net = net.saveobj;
    temp_net.Layers(4).Weights = abs(temp_net.Layers(4).Weights);
%     temp_net.Layers(4).Weights = (temp_net.Layers(4).Weights-0.5)*0.5+0.5;
%     temp_net.Layers(4).Bias = 0;
    temp_net.Layers(IR_LayerIdx).Weights = abs(temp_net.Layers(IR_LayerIdx).Weights);
%     temp_net.Layers(IR_LayerIdx).Weights = (temp_net.Layers(IR_LayerIdx).Weights-0.5)*0.5+0.5
%     temp_net.Layers(IR_LayerIdx).Bias = [0 0 0 0]';
    
    % train again with rectified weights
    rectified_net = net.loadobj(temp_net);
    lgraph2 = layerGraph(rectified_net)
    net = trainNetwork(Xtrain,Ytrain,lgraph2,options);
end

%% Analyze the weights of the trained network
net.Layers(4)
net.Layers(IR_LayerIdx).Weights
% net.Layers(IR_LayerIdx).Bias
[X,Y] = meshgrid(1:4,1:4);

figh = figure('position',[100 100 900 800]);
subplot(3,3,1)
imagesc(net.Layers(IR_LayerIdx).Weights); title('I->R weights')
tt = cellstr(num2str(round(net.Layers(IR_LayerIdx).Weights(:),2)));
text(X(:)-0.5,Y(:),tt,'FontSize',6)
ylabel('learned')
colorbar
caxis([0 1])

subplot(3,3,2)
imagesc(net.Layers(4).Weights); title('R->C weights')
colorbar
caxis([0 1])
% Visualize outputs
out = predict(net,Xtrain);

r2 = corrcoef(out,Ytrain);
err = mean((out-Ytrain).^2);
ax3 = subplot(3,3,3);
plot(Ytrain,out,'o'); hold on;
plot(0:0.01:1,0:0.01:1,'r','linewidth',2)
xlabel('desired')
ylabel('actual')
title(sprintf('r=%f',r2(1,2)))
annotStr = ['MSE: ' num2str(err)];
annotation('textbox','String',annotStr,'Position',ax3.Position,'Vert','top','FitBoxToText','on')

% Given ideal IR netcon weights, what does the network spit out?
targetIRNetcon = netcons.xrNetcon;
targetRCnetcon = netcons.rcNetcon';
% targetRCnetcon = [1 0 0 0]

% recfity weights
temp_net = net.saveobj;
% temp_net.Layers(4).Weights = targetRCnetcon;
% temp_net.Layers(4).Bias = 0;
temp_net.Layers(IR_LayerIdx).Weights = targetIRNetcon;
% temp_net.Layers(IR_LayerIdx).Bias = [0 0 0 0]';

subplot(3,3,4)
imagesc(temp_net.Layers(IR_LayerIdx).Weights); 
colorbar
caxis([0 1])

ylabel('Desired')
subplot(3,3,5)
imagesc(temp_net.Layers(4).Weights); 
colorbar
caxis([0 1])

% train again with rectified weights
ideal_net = net.loadobj(temp_net);
out = predict(ideal_net,Xtrain);

% plot "best case" vs training output
r2 = corrcoef(out,Ytrain);
err = mean((out-Ytrain).^2);
ax6 = subplot(3,3,6);
plot(Ytrain,out,'o'); hold on;
plot(0:0.01:1,0:0.01:1,'r','linewidth',2)
xlabel('desired')
ylabel('actual')
title(sprintf('r=%f',r2(1,2)))
xlim([0,1])
ylim([0,1])


% do linear regression - can get away with a simple Ax=b
m = out\Ytrain;
annotStr = {['MSE: ' num2str(err)];['m: ' num2str(m)]};
annotation('textbox','String',annotStr,'Position',ax6.Position,'Vert','top','FitBoxToText','on')

plot((0:0.01:1)*m,(0:0.01:1),'g','linewidth',2)

% cross correlation between weight petterns
r2 = corrcoef(targetIRNetcon(:),net.Layers(IR_LayerIdx).Weights(:))
subplot(3,3,4)
xlabel(['r = ' num2str(r2(1,2))])

rRC = corrcoef(targetRCnetcon(:),net.Layers(4).Weights(:))
subplot(3,3,5)
xlabel(['r = ' num2str(rRC(1,2))])

% Repeat for desired RC weights
% recfity weights
temp_net = net.saveobj;
temp_net.Layers(4).Weights = targetRCnetcon;
% temp_net.Layers(4).Bias = 0;
temp_net.Layers(IR_LayerIdx).Weights = targetIRNetcon;
% temp_net.Layers(IR_LayerIdx).Bias = [0 0 0 0]';

subplot(3,3,7)
imagesc(temp_net.Layers(IR_LayerIdx).Weights); 
colorbar
caxis([0 1])

ylabel('Desired')
subplot(3,3,8)
imagesc(temp_net.Layers(4).Weights); 
colorbar
caxis([0 1])

% run network again with rectified weights
ideal_net = net.loadobj(temp_net);
out = predict(ideal_net,Xtrain);

% plot "best case" vs training output
r2 = corrcoef(out,Ytrain);
err = mean((out-Ytrain).^2);

ax9 = subplot(3,3,9);
plot(Ytrain,out,'o'); hold on;
plot(0:0.01:1,0:0.01:1,'r','linewidth',2)
xlabel('desired')
ylabel('actual')
title(sprintf('r=%f',r2(1,2)))

% do linear regression - can get away with a simple Ax=b
m = out\Ytrain;
annotStr = {['MSE: ' num2str(err)];['m: ' num2str(m)]};
annotation('textbox','String',annotStr,'Position',ax9.Position,'Vert','top','FitBoxToText','on')
plot((0:0.01:1)*m,(0:0.01:1),'g','linewidth',2)
% Save plot
saveas(figh,['learned_results_set_' num2str(trainingSetNum) '.tif'])
% save net

%% save learned weights
learned.RCnetcon = net.Layers(4).Weights;
learned.IRnetcon = net.Layers(IR_LayerIdx).Weights;
save(sprintf('learnedWeights_set%02i.mat',trainingSetNum),'learned');


%% visualize psth - song only configs
% configsToPlot = [5,10,15,20];
% plotIdentifier = 'song only';
% plotPSTHandScatter;
%  
% configsToPlot = [6,12,18,24];
% plotIdentifier = 'colocated configs';
% plotPSTHandScatter;
% 
% configsToPlot = [7,13,19,22];
% plotIdentifier = 'mixed configs';
% plotPSTHandScatter;
