function [data,events,fs] = eegread(inFiles,fsOut,useLotsOfMemory)
% Function to read in data from (multiple) Brain Products .eeg files, 
% parsing the .vmrk and .vhdr files appropriately.
%

if(nargin < 1)
	error('Must input inFiles!');
end

if(nargin < 3 || isempty(useLotsOfMemory))
	useLotsOfMemory = false;
end

headers = {'DataFormat','DataOrientation','BinaryFormat','NumberOfChannels','SamplingInterval'};
headOuts = cell(length(headers),1);
markOffset = 0;

if ischar(inFiles)
	inFiles = {inFiles};
end

events = cell(length(inFiles),1);
data = cell(length(inFiles),1);
for ii = 1:length(inFiles)
	if(length(inFiles{ii})<4); error('Bad .eeg filename!'); end
	if(~strcmp(inFiles(1:end-3),'.eeg')); error('Bad .eeg filename!'); end
	if(~exist(inFiles{ii},'file')); error('File "%s" not found!',inFiles{ii}); end
	% First load in EEG header
	text = fileread(fullfile([inFiles{ii}(1:end-3) 'vhdr']));
	for hi = 1:length(headers)
		ind = regexp(text,headers{hi});
		if(numel(ind)~=1); error('Badness 10000'); end
		headOuts{hi} = strtok(text(ind+length(headers{hi})+1:end));
	end
	if(~strcmp(headOuts{1},'BINARY') || ~strcmp(headOuts{2},'MULTIPLEXED') || ~strcmp(headOuts{3},'IEEE_FLOAT_32')); error('Badness 10000'); end
	if(ii==1)
		fs = 1e6/str2double(headOuts{5});
		if(nargin < 2 || isempty(fsOut))
			fsOut = fs;
		end
		nCh = str2double(headOuts{4});
		if(isempty(fs) || isempty(nCh)); error('Badness 10000'); end
	else
		if(~isequal(fs,str2double(headOuts{5}))); error('Sample rate mismatch!'); end
		if(~isequal(nCh,str2double(headOuts{4}))); error('Channel count mismatch!'); end
	end

	% Now load in data, concatenating if necessary
	fid = fopen(inFiles{ii});
	if(~useLotsOfMemory) % Conserve memory by doing lots of disk IO
		for ci = 1:nCh
			% Move 4 bytes (for float32) from the start of the file for each channel
			fseek(fid,4*(ci-1),-1);
			tempData = fread(fid,inf,'float32',4*(nCh-1)); % Skip nCh-1 float 32's after each read
			if(fsOut~=fs)
				temp = resample(tempData,fsOut,fs);
			else
				temp = tempData;
			end
			if(ci==1)
				tempNsRe = length(temp);
				tempDataRe = zeros(tempNsRe,nCh);
			end
			tempDataRe(:,ci) = temp;
		end
	else % Burn memory, but less disk IO
		tempData = fread(fid,inf,'*float32'); % Read in the whole thing in float32 format
		inds = 0:nCh:length(tempData)-1;
		for ci = 1:nCh
			if(fs~=fsOut)
				temp = resample(double(tempData(inds+ci)),fsOut,fs);
			else
				temp = double(tempData(inds+ci));
			end				
			if(ci==1)
				tempNsRe = length(temp);
				tempDataRe = zeros(tempNsRe,nCh);
			end
			tempDataRe(:,ci) = temp;
		end
	end
	fclose(fid);
	data{ii} = tempDataRe;

	% Parse marker file
	text = fileread([inFiles{ii}(1:end-3) 'vmrk']);
	% CLEAN UP STIMULUS/RESPONSE!
	eves = regexp(text,'Mk\d+=[SR][te][is][mp][uo][ln][us][se],[SR][ ]{1,2}\d{1,2},\d+,1,0');
	tempEvents = zeros(length(eves),2);
	for ei = 1:length(eves)
		[temp,residual] = strtok(text(eves(ei):end),'=');
		if(ei+1~=str2double(temp(3:end))); error('Bad marker number'); end
		sind = regexp(residual,',[SR]','ONCE');
		if(isequal(residual(sind+1),'S'))
			offset = 1;
		elseif(isequal(residual(sind+1),'R'))
			offset = 16;
		else
			error('Could not determine Stimulus or Response!');
		end
			
		tind = regexp(residual,'\d{1,2},\d+,1,0','ONCE');
		[temp,residual] = strtok(residual(tind:end),',');
		if(isempty(str2double(temp))); error('Badness 10000'); end
		tempEvents(ei,2) = offset*str2double(temp);
		temp = strtok(residual,',');
		if(isempty(str2double(temp))); error('Badness 10000'); end
		tempEvents(ei,1) = str2double(temp);
	end
	events{ii} = [round(tempEvents(:,1)*(fsOut/fs))+markOffset tempEvents(:,2)];
	markOffset = markOffset + tempNsRe;
end
data = vertcat(data{:});
events = vertcat(events{:});
fs = fsOut;
