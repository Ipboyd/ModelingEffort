function matlabpoolPrep(Nworkers)

if exist('matlabpool')
	Ncurrent = matlabpool('size');
	if(nargin<1)
		if(~Ncurrent)
			matlabpool open local;
		end
	else
		if(~Ncurrent)
			matlabpool('open','local',Nworkers);
		elseif(Ncurrent ~= Nworkers)
			matlabpool close;
			matlabpool('open','local',Nworkers);
		end
	end
else
	disp('MATLABPOOLPREP: Parallel Toolbox not installed.')
end