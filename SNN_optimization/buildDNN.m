%% Prep data
load('training_set_1.mat','output_training','input_training')

trainBatch = 0.9;

m = size(input_training,1)
m_train = randsample(m,round(m*trainBatch));
Xtrain = input_training(m_train,:);
Ytrain = output_training(m_train,:);
Xtest = input_training;
Xtest(m_train,:) = [];
Ytest = output_training;
Ytest(m_train,:) = [];
Create Layers
%% define main path
mainPath = [
    featureInputLayer(4,'Name','input')
    
    additionLayer(2,'Name','add_In_Inh')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(1,'Name','C')
    regressionLayer('Name','output')
    ]

inhPath = [
    negationLayer('negation')
    fullyConnectedLayer(4,'Name','R')
    ]

layer graph of main path
lgraph = layerGraph(mainPath)
plot(lgraph)


%% Add inh path to main path
lgraph = addLayers(lgraph,inhPath)
lgraph = connectLayers(lgraph,'input','negation')
lgraph = connectLayers(lgraph,'R','add_In_Inh/in2')

plot(lgraph)


%% Analyze network
analyzeNetwork(lgraph)

%% Train
maxEpochs = 100;
miniBatchSize = 1024;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'ValidationData',{Xtest,Ytest}, ...
    'ValidationFrequency',30, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(Xtrain,Ytrain,lgraph,options)


%% Check network weights
net.Layers

i = 7  %choose appropriate index to get the desired layer
net.Layers(i).Weights




%% References
% getting started: https://www.mathworks.com/help/deeplearning/ug/train-a-convolutional-neural-network-for-regression.html
% layers: https://www.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
% resnet example: https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html
% analyze network architecture: https://www.mathworks.com/help/deeplearning/ref/analyzenetwork.html