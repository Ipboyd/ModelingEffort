function [strf,tvec,fvec,uncert,strfJN,tol,cc,predInfo,psth_est,ccNL,predInfoNL] = makeBoost(spikeTimes,stims,NBAND,fsSTRF,tol,flims,winlen)
%MAKEBOOST Make a STRF from recorded spike times and stimuli.
%   S = MAKEBOOST(ST,STIMS) will generate a STRF S using boosting
%   routines using the cell matrix of spike times SPIKETIMES and the 
%   one-channel stimulus .wav files (that must be named "stim_1.wav", 
%   "stim_2.wav", ...) in the full path given by STIMS. STIMS can also be a
%   cell array of audio data (1xN or Nx1) that will be converted to a
%   spectrogram.
%
%   S = MAKEBOOST(ST,STIMS,NUMBAND) calculate a STRF using the specified
%   number of frequency bands NBAND (default=64 bands).
%
%   S = MAKEBOOST(ST,STIMS,[],FS) or MAKEBOOST(ST,STIMS,NBAND,FS) will
%   calculate the STRF using the specified sample rate for the spike times
%   (default=1000Hz).
%
%   S = ... or MAKEBOOST(ST,STIMS,NBAND,FS,TOLS) will calculate the STRF
%   using the specified vector of iterations (default [80 120 160 200 240
%   320 400 480 560 640 720 800 960]). Note: if only one tolerance is given
%   and 3 or fewer output arguments (s,t,f only) are requested, the 
%   prediction and validation steps are skipped (some speed savings).
%
%   S = ... or MAKEBOOST(ST,STIMS,NBAND,FS,TOLS,FLIMS) will calculate the
%   STRF using NBANDS in the region of frequencies specified by the
%   two-element vector FLIMS.
%
%   S = ... or MAKEBOOST(ST,STIMS,NBAND,FS,TOLS,FLIMS,WINLEN) will 
%   calculate the STRF using the window limits [WINLEN(1) WINLEN(2)] or
%   [-WINLEN(1) WINLEN(1)] (in seconds) for 2- or 1-element vector WINLEN.
%
%   [S,TVEC,FVEC,UNCERT,JN,TOL,CC] = MAKEBOOST(...) also returns the time 
%   and frequency vectors, the CC of the unit, the tolerance that yielded
%   that CC, and the UNCERT uncertainty in the STRF estimate (based on a
%   jackknife estimate of JN strfs.
%
%   [S,TVEC,FVEC,UNCERT,JN,TOL,CC,PREDINFO,OFFSET] = MAKEBOOST(...) also
%   returns the PREDINFO array. This also causes TOL and CC to become
%   vectors corresponding to those PREDINFO values, and S, JN, and UNCERT
%   to become arrays of matrices computed for each tolerance (last index).
%   OFFSET is the psth offset value (constant).
%   

if(nargin < 3 || isempty(NBAND))
    NBAND = 64;
end
if(nargin < 4 || isempty(fsSTRF))
    fsSTRF = 500;
end
if(nargin < 5 || isempty(tol))
    tol = 160:160:3200;%[80 160 240 400 560 720 960 1280 1600 1920 2000 2160 2320 2480 2640 2800 3000];
end
if(nargin < 6 || isempty(flims))
    flims = [250 8000];
end
if(nargin < 7 || isempty(winlen))
    winlen = [-0.01 0.1];
end
outputMaxOnly = nargout < 8;

%% Generate stimulus and response files
[stims,resps,fvec] = makeSTRFhelper(stims,spikeTimes,NBAND,fsSTRF,flims);
for ii = 1:size(length(resps))
    stims{ii} = single(stims{ii});
    resps{ii} = single(resps{ii});
end

%% Run actual STRF calculation: calcSTRF does everything in samples
winlen = round(winlen*fsSTRF); % 100 ms window by default
if(nargout > 8)
    [strf,uncert,strfJN,cc,predInfo,psth_est,ccNL,predInfoNL] = calcBoost(stims,resps,winlen,tol);
else
    [strf,uncert,strfJN,cc,predInfo] = calcBoost(stims,resps,winlen,tol);
end        
if(outputMaxOnly)
    [junk,ind] = max(predInfo); %#ok<ASGLU>
    cc = cc(ind);
    strfJN = strfJN(:,:,:,ind);
    uncert = uncert(:,:,ind);
    strf = strf(:,:,ind);
    tol = tol(ind);
    if(nargout > 8)
        psth_est = psth_est(:,:,ind);
        ccNL = ccNL(ind);
        predInfoNL = predInfoNL(ind);
    end
end
predInfo = predInfo*fsSTRF; % Convert predInfo from info/sample to info/time
tvec = (winlen(1):winlen(2))/fsSTRF;
