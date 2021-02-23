trainingSetNum = 8;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 9;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 10;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;
netcons.xrNetcon(4,1) = 1;

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 11;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;
netcons.xrNetcon(4,1) = 1;
netcons.xrNetcon(2,4) = 1;

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

for trainingSetNum = 12:15
	netcons.xrNetcon = round(rand(4)); % cross channel inhibition
	genTrainingData;
	disp(['finished ' num2str(trainingSetNum)])
end
