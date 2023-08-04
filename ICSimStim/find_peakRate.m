function [allTS, env, peakDrv, peakEnv] = find_peakRate(sound, soundfs, soundinputtype)
% function [allTS, env, peakRate, peakEnv] = find_peakRate(sound, soundfs, onsOff, envtype)
% Inputs: 
%   sound - time x 1, sound waveform
%   soundfs - 1x1, sampling frequency of sound
%   [OBS]onsOff  - 1 x 2 times of stimulus onset and offset in sound (in seconds) 
%   envtype: 'rms' (default), or 'broadband': specific loudness envelope or broadband envelope
% Output: 
%   allTS - timestamps of all landmarks of interest
%   env - amplitude envelope of input
%   peakDrv - discrete time series of peakRate events in envelope
%   peakEnv - discrete time series of peakEnv events
%
% KP, 2019; updated 2023 


%% initialize
if nargin<3, soundinputtype = 'amp_dB'; end
% if nargin<4, envtype = 'broadband'; end

% dB CHANGE THRESHOLD
min_peak_incr = 6; 

% LANDMARK CLEANUP FLAG
% if set to 1, landmark series will be cleaned up to contain only a single
% peakRate event in each envelope cycle, defined as envelope trough-to-trough
cleanup_flag = 1;


%% get envelope

envfs    = 1000;
filtprop = envfs/soundfs/20;

switch soundinputtype  
    case 'broadband' %% broadband envelope        
        rectsound  = abs(sound);
        [b,a] = butter(4, filtprop); %10/(soundfs/2)
        cenv = filtfilt(b, a, rectsound);
        downsenv = resample(cenv, (1:length(cenv))/soundfs, envfs);
        downsenv(downsenv <0) =0;
        env = downsenv;
    case 'amp_linear'
        env = sound2dB(sound);
    case 'amp_dB'
        env = sound;
end
mpd = 1000/soundfs; % 1 ms - convert to env fs if not 1000


%% get landmarks in envelope

allTS    = find_landmarks(env, cleanup_flag, mpd, min_peak_incr); 
peakEnv  = allTS(4,:);
peakDrv  = allTS(6,:);


% figure; hold on
% plot(env)
% plot(peakRate)
% plot(diff(env))

% figure; 
% histogram(peakRate,logspace(-3,-1,100))
% ylim([0 15])
% xlim([1e-3 1e-1])
% set(gca,'xscale','log')

end



%% landmark detection in envelope
function [allTS, varNames] = find_landmarks(TS, cleanup_flag, mpd, min_peak_incr)

%% find discrete events in envelope

% make sure envelope is row vector
if ~isrow(TS)
    TS = TS';
end

% Bring max envelope to 0
TS = TS-max(TS)-0.01;

envFloor = ceil(min(TS)+1);


%% discrete loudness

% min
% [lmin, minloc] = findpeaks(-TS,'MinPeakDistance',mpd);
[lmin, minloc] = findpeaks(-TS,'MinPeakProminence',min_peak_incr);
minEnv = zeros(size(TS));
minEnv(minloc)=-lmin;

% max
% [lmin, minloc] = findpeaks(TS,'MinPeakDistance',mpd);
[lmin, minloc] = findpeaks(TS,'MinPeakProminence',min_peak_incr);
peakEnv = zeros(size(TS));
peakEnv(minloc)=lmin;



% Depending on  stimuli, if there are longer silent ITIs, may need to add
% something to ensure that minEnv event occurs at/near stimulus onset 

% Check for cases where two mins without a max in between them 
Mins = find(minEnv);
Maxs = find(peakEnv);

for im = 2:numel(Mins)
    
    iMax = find(Maxs>Mins(im-1) & Maxs<Mins(im));
    
    mE = envFloor-1;
    if isempty(iMax)
        [mE,imE] = max(TS(Mins(im-1):Mins(im)));
    end
    
    % Include only if envelope exceeds a low threshold
    if mE>envFloor+2
        peakEnv(imE+Mins(im-1)-1) = mE;
    end
end


%% discrete delta loudness

% first temporal derivative of TS
diff_loudness = [0 diff(TS)];

% min
negloud = diff_loudness; negloud(negloud>0) = 0;
[lmin, minloc] = findpeaks(-negloud,'MinPeakDistance',mpd);
decrDrv = zeros(size(TS));
decrDrv(minloc)=-lmin;

% max
posloud = diff_loudness; posloud(posloud<0) = 0;
[lmin, minloc] = findpeaks(posloud,'MinPeakDistance',mpd);
peakDrv = zeros(size(TS));
peakDrv(minloc)=lmin;

clear negloud posloud 


%% ------- KP clean up 
% for each minEnv to maxEnv, keep the highest peakRate event

if cleanup_flag

Mins = find(minEnv);
Maxs = find(peakEnv);
iprs = find(peakDrv);

NewPeakDrv = zeros(size(peakDrv));

for im = 1:numel(Mins)
    
    % Find beginning and end of this level ramp
    t1 = Mins(im);
    t2 = Maxs(find(Maxs>t1,1,'first'));
    if isempty(t2)
        continue
    end
    
    % Find maximum peak of derivative
    thesePRs = iprs(iprs>t1 & iprs<t2);
    [maxRise,iRise] = max(peakDrv(thesePRs));
    
    % Set all other peakRate events to keep
    NewPeakDrv(thesePRs(iRise)) = maxRise;
    
end
end

% % % Plot to check
% figure;
% plot(TS,'k','LineWidth',2)
% hold on
% % plot(find(peakRate),TS(1,peakRate~=0),'.r','MarkerSize',10)
% plot(find(peakEnv),TS(1,peakEnv~=0),'.g','MarkerSize',20)
% plot(find(minEnv),TS(1,minEnv~=0),'.c','MarkerSize',20)

% plot(find(NewPeakRate),TS(NewPeakRate~=0),'*b','MarkerSize',20)


%% Finish storing output information

allTS = [TS; ...
    diff_loudness;...
    minEnv;...
    peakEnv;...
    decrDrv;...
    NewPeakDrv];

varNames = {'Loudness', 'dtLoudness', 'minEnv', 'peakEnv', 'minDrv', 'peakDrv'};

end
