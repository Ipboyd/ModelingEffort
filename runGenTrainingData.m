trainingSetNum = 2;
netcons.xrNetcon = zeros(4);
netcons.rcNetcon = [1 0 1 0]';
genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 8;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 9;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;
netcons.rcNetcon = [1 1 1 1]';

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 10;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;
netcons.xrNetcon(4,1) = 1;
netcons.rcNetcon = [1 1 1 1]';

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

trainingSetNum = 11;
netcons.xrNetcon = zeros(4); % cross channel inhibition
netcons.xrNetcon(2,1) = 1;
netcons.xrNetcon(3,1) = 1;
netcons.xrNetcon(4,1) = 1;
netcons.xrNetcon(2,4) = 1;
netcons.rcNetcon = [1 1 1 1]';

genTrainingData;
disp(['finished ' num2str(trainingSetNum)])

xrNetconMask = ones(4)-eye(4);
netcons.rcNetcon = [1 1 1 1]';
for trainingSetNum = 12:15
	netcons.xrNetcon = round(rand(4)) .* xrNetconMask; % cross channel inhibition
	genTrainingData;
	disp(['finished ' num2str(trainingSetNum)])
end

xrNetconMask = ones(4)-eye(4);
netcons.rcNetcon = rand(4,1);
for trainingSetNum = 16:19
	netcons.xrNetcon = rand(4) .* xrNetconMask; % cross channel inhibition
	genTrainingData;
	disp(['finished ' num2str(trainingSetNum)])
end