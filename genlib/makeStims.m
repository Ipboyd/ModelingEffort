function [stimMat reorder] = makeStims(fnames,Ntrials,Tdelay,fs,outname,reconCheck)

if(nargin < 6 || isempty(reconCheck))
    reconCheck = 0;
end

Nsongs = length(fnames);

% Read in stimulus files
stim0Mat = cell(Nsongs,1);
for ci = 1:Nsongs
    stim0Mat{ci} = datread(fnames{ci});
end; clear ci temp;
Nssongs = cellfun('length',stim0Mat);
Nsdelay = ceil(Tdelay*fs);
Nstot = sum(Nssongs) + Nsdelay*(Nsongs-1);
fprintf('Generating %d trials for %d songs ==> ~%s run\n',Ntrials,Nsongs,timestring((Nstot*Ntrials)/fs))

stimMat = zeros(Nstot,Ntrials); % Store stimuli to be played
ordMat = zeros(Nsongs,Ntrials);
for ti = 1:Ntrials
    ordMat(:,ti) = randperm(Nsongs);
    offset = 0;
    for oi = 1:Nsongs
        ci = ordMat(oi,ti);
        stimMat(offset+(1:Nssongs(ci)),ti) = stim0Mat{ci};
        offset = offset + Nssongs(ci) + Nsdelay;
    end
    if(offset-Nsdelay ~= Nstot)
        error('Incorrect stimulus calculation');
    end
end; clear ti ci;

% Check to make sure we can appropriately re-constitute responses
if(reconCheck)
    fprintf('Checking reconstitutability...\n')
    % Generate fake ideal responses
    respMat = zeros(Nstot,Ntrials);
    for ti = 1:Ntrials
        offset = 0;
        for oi = 1:Nsongs
            ci = ordMat(oi,ti);
            respMat(offset+(1:Nssongs(ci)),ti) = ci;
            offset = offset + Nssongs(ci) + Nsdelay;
        end
        if(offset-Nsdelay ~= Nstot)
            error('Incorrect stimulus calculation');
        end
    end; clear ti ci;

     % What would be the reconstituted matrix
    rordMat = zeros(max(Nssongs),Ntrials,Nsongs);
    for ti = 1:Ntrials
        offset = 0;
        for oi = 1:Nsongs
            ci = ordMat(oi,ti);
            rordMat(1:Nssongs(ci),ti,ci) = respMat(offset+(1:Nssongs(ci)),ti);
            offset = offset + Nssongs(ci) + Nsdelay;
        end
        if(offset-Nsdelay ~= Nstot)
            error('Incorrect stimulus calculation');
        end
    end

    wrongMat = zeros(Nsongs,Ntrials);
    for ci = 1:Nsongs
        for ti = 1:Ntrials
            wrongMat(ci,ti) = any(rordMat(1:Nssongs(ci),ti,ci) ~= ci) || any(rordMat(Nssongs(ci)+1:max(Nssongs),ti,ci) ~= 0);
        end
    end; clear ci ti;
    if(any(wrongMat))
        error('Could not properly reconstitute');
    end
    fprintf('\bdone\n')
end; clear rordMat respMat wrongMat offset ci ti oi;

reorder.ordMat = ordMat;
reorder.Nsdelay = Nsdelay;
reorder.Nssongs = Nssongs;

innie = '';
if(isdir(outname))
    innie = lower(input(sprintf('Directory "%s" exists: overwrite files Y/N [N]? ',outname),'s'));
    if isempty(innie)
        innie = 'N';
    end
end

if(~isdir(outname) || innie(1) == 'y' || innie(1) == 'Y')
    if(~isdir(outname))
        mkdir(outname);
    end
    fprintf('Writing files...\n')
    for ti = 1:Ntrials
        datwrite(stimMat(:,ti),sprintf('%s/%s%03.0f.dat',outname,outname,ti));
    end
    eval(sprintf('save %s/%sreorder reorder',outname,outname));
    fid = fopen(sprintf('%s/%s.txt',outname,outname),'w');
    fprintf(fid,sprintf('D:\\\\data\\\\stimuli\\\\%s\\\\%s%%03.0f.dat\r',outname,outname),1:Ntrials);
    fclose(fid); clear fid;
    fprintf('\bdone\n')
    fprintf('Elapsed time will be %s\n',timestring(Nstot*Ntrials/fs))
else
    disp('No files outputted');
end
