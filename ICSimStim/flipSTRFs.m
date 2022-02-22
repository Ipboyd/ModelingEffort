folders = dir(uigetdir(fullfile(cd,'ICSimStim','mouse','full_grids')));
folders(~contains({folders.name},'s')) = [];

for f = 1:length(folders)
    allfiles = dir([folders(f).folder filesep folders(f).name]);
    matfiles = allfiles(contains({allfiles.name},'.mat'));
    
    newfolder = [extractBefore(allfiles(f).folder,'_202') '-fixed'];
    mkdir(newfolder);
    
    % fix matfiles
    for m = 1:length(matfiles)
        strffile = load([matfiles(m).folder filesep matfiles(m).name]);
        
        % flip song and masker locations
        if strffile.songloc ~= 0
            strffile.songloc = 5-strffile.songloc;
        end
        
        if strffile.maskerloc ~= 0
            strffile.maskerloc = 5-strffile.maskerloc;
        end
        
        % flip t_spiketimes
        strffile.t_spiketimes = strffile.t_spiketimes(:,[4:-1:1 8:-1:5]);
        
        % flip avgSpkRate
        strffile.avgSpkRate = fliplr(strffile.avgSpkRate);
        
        % flip disc
        strffile.disc = fliplr(strffile.disc);
        
        % flip fr
        strffile.fr = strffile.fr(:,[4:-1:1 8:-1:5]);
        
        newname = ['s' num2str(strffile.songloc) 'm' num2str(strffile.maskerloc) '.mat'];
        
        save(fullfile(newfolder,newname),'-struct','strffile');
    end
    
    % for tiffs
    tiffs = allfiles(contains({allfiles.name},'.tiff'));
    for t = 1:length(tiffs)
        if ~strcmp(tiffs(t).name(2),'0')
            songloc = 5-str2double(tiffs(t).name(2));
        else
            songloc = 0;
        end
        if ~strcmp(tiffs(t).name(4),'0')
            maskloc = 5-str2double(tiffs(t).name(4));
        else
            maskloc = 0;
        end
        
        copyfile(fullfile(tiffs(t).folder,tiffs(t).name) , ...
            fullfile(newfolder,['s' num2str(songloc) 'm' num2str(maskloc) '_unfixed.tiff']));
    end
    
    copyfile(fullfile(allfiles(f).folder,'performance_grid.tiff'),...
        fullfile(newfolder,'performance_grid_unfixed.tiff'));
    
    copyfile(fullfile(allfiles(f).folder,'notes.txt'),...
        fullfile(newfolder,'notes.txt'));
    
end