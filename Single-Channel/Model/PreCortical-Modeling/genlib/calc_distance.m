% input mat, tau (ms), fs
function distmat = calc_distance(mat, tau, fs, stretches)

% This has now been redone again to use analytical integration, which is as
% fast as it could possibly get.
% RKM, Oct 2009

if nargin < 4
	distmat = calcvr(sm2st(mat, fs), tau/1e3);
else
	% This has now been redone to use expconv, which is much faster than everything.
	% RKM, July 2008
	
	[len numtrials numstims] = size(mat);
	if nargin < 4
		stretches = ones(1, numstims);
	end
	tau = tau/1000;
	numTaus = length(tau);
	
	distmat = zeros([numtrials*numstims*[1 1], length(tau)]);
	
	% extend the matrix by 5 time constants
	mat = double(mat);
	s = size(mat);
	mat = cat(1, mat, zeros([round(max(tau)*fs*5), s(2:end)]));
	
	% initialize y so filtering only has to be done once
	y = zeros(size(mat));
	
	for tauNum = 1:numTaus
		% filter y with the exponentials
		for stim = 1:numstims
			y(:,:,stim) = expconv(mat(:,:,stim), tau(tauNum)*stretches(stim), fs);
		end
		
		% compute the distances
		numrows = numstims*numtrials;
		for row = 1:numrows - 1
			for col = row + 1:numrows
				distmat(row,col,tauNum) = sum((y(:,row) - y(:,col)).^2)/fs;
			end
		end
		
		distmat(:,:,tauNum) = (distmat(:,:,tauNum) + distmat(:,:,tauNum).')/tau(tauNum);
	end
end