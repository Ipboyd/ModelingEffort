function [strf,tvec,fvec,uncert,strfJN,tol,cc,predInfo,psth_est,ccNL,predInfoNL] = makeSTRF(spikeTimes,stims,NBAND,fsSTRF,tol,flims,winlen,forceCausal,forceWindow,doRealignment)
%MAKESTRF Make a STRF from recorded spike times and stimuli.
%   S = MAKESTRF(ST,STIMS) will generate a STRF S using STRFPAK-derived
%   routines using the cell matrix of spike times SPIKETIMES and the 
%   one-channel stimulus .wav files (that must be named "stim_1.wav", 
%   "stim_2.wav", ...) in the full path given by STIMS. STIMS can also be a
%   cell array of audio data (1xN or Nx1) that will be converted to a
%   spectrogram.
%
%   S = MAKESTRF(ST,STIMS,NUMBAND) calculate a STRF using the specified
%   number of frequency bands NBAND (default=64 bands).
%
%   S = MAKESTRF(ST,STIMS,[],FS) or MAKESTRF(ST,STIMS,NBAND,FS) will
%   calculate the STRF using the specified sample rate for the spike times
%   (default=1000Hz).
%
%   S = ... or MAKESTRF(ST,STIMS,NBAND,FS,TOLS) will calculate the STRF
%   using the specified vector of tolerances (default [0.05 0.01 0.005
%   0.0025 0.001 0.0005 0.00025 0.0001 0.00005]). Note: if only one
%   tolerance is given and 3 or fewer output arguments (s,t,f only) are
%   requested, the prediction and validation steps are skipped (large speed
%   savings).
%
%   S = ... or MAKESTRF(ST,STIMS,NBAND,FS,TOLS,FLIMS) will calculate the
%   STRF using NBANDS in the region of frequencies specified by the
%   two-element vector FLIMS.
%
%   S = ... or MAKESTRF(ST,STIMS,NBAND,FS,TOLS,FLIMS,FORCECAUSAL) will
%   multiply the STRF by a causal window if the FORCECAUSAL is true.
%   (Default: true, force causality.)
%
%   S = ... or MAKESTRF(ST,STIMS,NBAND,FS,TOLS,FLIMS,WINLEN,FCAUSAL)
%   will multiply the STRF by a causal window if the FCAUSAL is true.
%   (Default: true, force causality.)
%
%   S = ... or MAKESTRF(ST,STIMS,NBAND,FS,TOLS,FLIMS,WINLEN,FCAUSAL,FWINDOW)
%   will multiply the STRF by a hanning window prior to SVD decomposition
%   if the FWINDOW is true. (Default: true, force windowing.)
%
%   [S,TVEC,FVEC,UNCERT,JN,TOL,CC] = MAKESTRF(...) also returns the time 
%   and frequency vectors, the CC of the unit, the tolerance that yielded
%   that CC, and the UNCERT uncertainty in the STRF estimate (based on a
%   jackknife estimate of JN strfs.
%
%   [S,TVEC,FVEC,UNCERT,JN,TOL,CC,PREDINFO] = MAKESTRF(...) also returns 
%   the PREDINFO array. This also causes TOL and CC to become vectors 
%   corresponding to those PREDINFO values, and S, JN, and UNCERT to become
%   arrays of matrices computed for each tolerance (last index).
%

if(nargin < 3 || isempty(NBAND))
    NBAND = 64;
end
if(nargin < 4 || isempty(fsSTRF))
    fsSTRF = 1000;
end
if(nargin < 5 || isempty(tol))
    tol = [0.05 0.01 0.005 0.0025 0.001 0.0005 0.00025 0.0001 0.00005];%[0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005];
end
if(nargin < 6 || isempty(flims))
    flims = [250 8000];
end
if(nargin < 7 || isempty(winlen))
    winlen = 0.1;
end
if(nargin < 8 || isempty(forceCausal))
    forceCausal = true;
end
if(nargin < 9 || isempty(forceWindow))
    forceWindow = true;
end
if(nargin < 10 || isempty(doRealignment))
    doRealignment = true;
end

runPredVal = nargout > 6 || length(tol) > 1;
outputMaxOnly = nargout < 8;

%% Generate stimulus and response files
[stims,resps,fvec] = makeSTRFhelper(stims,spikeTimes,NBAND,fsSTRF,flims);

%% Run actual STRF calculation: calcSTRF does everything in samples
winlen = round(winlen*fsSTRF); % 100 ms window by default
if(runPredVal)
    if(nargout > 8)
        [strf,uncert,strfJN,cc,predInfo,psth_est,ccNL,predInfoNL] = calcSTRF(stims,resps,winlen,tol,forceCausal,forceWindow,doRealignment);
    else
        [strf,uncert,strfJN,cc,predInfo] = calcSTRF(stims,resps,winlen,tol,forceCausal,forceWindow,doRealignment);
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
else
    [strf,uncert,strfJN] = calcSTRF(stims,resps,winlen,tol,forceCausal,forceWindow,doRealignment);
end
tvec = (-winlen:winlen)/fsSTRF;
