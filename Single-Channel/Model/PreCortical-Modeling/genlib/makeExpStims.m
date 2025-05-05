function makeExpStims(iniPath,stimPath,outputPath)

% Can leave stimPath empty if on the real computer.

doMono = true;

ini=ini2struct(iniPath);

if(nargin < 3 || isempty(stimPath))
	[stimPath,junk] = fileparts(ini.methods.file1); %#ok<NASGU>
end
if(~isdir(outputPath))
	mkdir(outputPath);
end

nStimuli = str2double(ini.methods.nstimuli);
fs = str2double(ini.methods.samplerate);
nOutChannels = str2double(ini.methods.noutchannels);

%% Generate stimulus files
for stimNum = 1:nStimuli
	% Convert and save stimulus
	stimLen = eval(sprintf('str2double(ini.results.stim%ilen)', stimNum));
	stimList = eval(sprintf('ini.methods.stim%d',stimNum));
	stimLims = [0 strfind(stimList,':') length(stimList)+1];
	if(length(stimLims)-1 ~= nOutChannels)
		error('Mismatch between stimulus %d and noutchannels=%d',stimNum,nOutChannels)
	end
	stim = zeros(stimLen,nOutChannels);
	for ci = 1:nOutChannels
		fileString = stimList(stimLims(ci)+1:stimLims(ci+1)-1);
		fileLims = [0 strfind(fileString, '+') length(fileString)+1];
		nFiles = length(fileLims) - 1;
		fileNum = zeros(nFiles, 1);
		for fileInd = 1:nFiles
			fileNum(fileInd) = str2double(fileString(fileLims(fileInd)+1:fileLims(fileInd+1)-1));
			% fileNum = str2double(stimList(stimLims(ci)+1:stimLims(ci+1)-1));
			if(fileNum(fileInd) > 0)
				[junk, tempFile] = fileparts(replaceslashes(eval(sprintf('ini.methods.file%d',fileNum(fileInd))))); %#ok<ASGLU>
				tempFile = fullfile(stimPath,tempFile);
				tempWav = wavread(sprintf('%s.wav', tempFile));
				stim(1:length(tempWav),ci) = stim(1:length(tempWav),ci) + tempWav;
			end
		end
	end
	if(doMono)
		stim = mean(stim,2);
	end
	stimFile = fullfile(outputPath,sprintf('stim_%d.wav',stimNum));
	wavwrite(stim,fs,stimFile);
	if stimNum == 520
% 		keyboard
	end
end

end

function x = replaceslashes(x)
x(x == '\' & [x(2:end) 0] == '\') = [];
x(x == '\') = '/';
end