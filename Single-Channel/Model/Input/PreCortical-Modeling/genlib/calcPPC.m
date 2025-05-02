function y = calcPPC(phase)
% Calculates the PPC (~PLV^2) along the rows of PHASES. Mex the function
% "calcPPChelper.c" for a substantial speedup.

if(exist('calcPPChelper') == 3)
	phase = phase.';
	cosPhase = cos(phase);
	sinPhase = sin(phase);
	y = calcPPChelper(cosPhase,sinPhase);
else
	warning('Need to mex calcPPChelper.c! Using 50% slower MATLAB code.');
	cosPhase = cos(phase);
	sinPhase = sin(phase);
	pairs = nchoosek(1:size(phase,2), 2);
	pair1 = pairs(:,1);
	pair2 = pairs(:,2);
	y = zeros(size(phase,1),1);
	nPairs = size(pairs,1);
	for pairNum = 1:nPairs
		% use a trig identity to do this: y = y + cos(diff(angle(fft(x(:,pairs(pairNum,))), [], 2));
		y = y + cosPhase(:,pair1(pairNum)).*cosPhase(:,pair2(pairNum)) + sinPhase(:,pair1(pairNum)).*sinPhase(:,pair2(pairNum));
	end
	y = y/nPairs;
end
