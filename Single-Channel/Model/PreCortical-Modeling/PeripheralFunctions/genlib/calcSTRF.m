function [strf,uncert,strfJN,cc,predInfo,psth_est,ccNL,predInfoNL] = calcSTRF(stims,resps,winlen,tol,forceCausal,forceWindow,doRealignment)

doNL = false;
if(nargin<5 || isempty(forceCausal))
    forceCausal = true;
end
if(nargin<6 || isempty(forceWindow))
    forceWindow = true;
end
if(nargin<7 || isempty(doRealignment))
    doRealignment = true;
end
if(nargout > 5)
    doNL = true;
end

runPredVal = nargout > 3;
ntol = length(tol);

%% 1. Estimation
nstim = length(stims);
nlens = cellfun(@size,stims,repmat({1},nstim,1));
if(~isequal(nlens,cellfun(@size,resps,repmat({1},length(resps),1))))
    error('Stimulus and response lengths (columns) must match');
end
NBAND = unique(cellfun(@size,stims,repmat({2},nstim,1)));
if(numel(NBAND) > 1)
    error('Stimuli must have the same number of rows');
end
ntrialss = cellfun(@size,resps,repmat({2},nstim,1));

ntvec = 2*winlen + 1;   % temporal axis range
tinds = -winlen:winlen;

% Determine the average stimulus value in each band across all stimuli
a=cellfun(@sum,stims,repmat({1},nstim,1),'UniformOutput',false);
stim_avg = sum(bsxfun(@times,vertcat(a{:}),ntrialss),1)/(nlens.'*ntrialss);
psth_avg = mean(vector(vertcat(resps{:})));

w = hanning(ntvec).';
if(~forceWindow)
    w = ones(size(w));
end

% cal_AutoCorr territory: Stimulus autocorrelation

CS = zeros(nchoosek(NBAND+1,2), ntvec);
nyqind = (ntvec-1)/2 + 1;
inds = [nyqind:ntvec 1:nyqind-1];
stimval = cell(nstim,1);
for k = 1:nstim
    stimval{k} = bsxfun(@minus,stims{k},stim_avg);
    CS_diff = zeros(nchoosek(NBAND+1,2),ntvec);
    for ii=1:ntvec
        tid = tinds(ii);
        onevect = (max(1,tid+1)):(min(nlens(k),nlens(k)+tid));
        temp = stimval{k}(onevect,:).'*stimval{k}(onevect - tid,:);
        CS_diff(:,ii) = temp(tril(true(NBAND))).';
    end
    CS = CS + CS_diff*ntrialss(k);
end
CS = bsxfun(@times,CS(:,inds),w(inds));
CS_ns = sum(bsxfun(@times,bsxfun(@minus,nlens,abs(tinds(inds))),ntrialss),1);
CSf = fft(bsxfun(@rdivide,CS,(CS_ns==0) + CS_ns),[],2);

% cal_CrossCorr territory: Spike crosscorrelation

CSR_JN = zeros(NBAND,ntvec,nstim);
for k = 1:nstim
    stimval{k} = flipud(stimval{k});
    thisResp =  flipud(mean(resps{k},2) - psth_avg);
    % Time-reversed xcov between response and stimulus in each band
    for ib1 = 1:NBAND
        CSR_JN(ib1,:,k) = xcorr(stimval{k}(:, ib1), thisResp, winlen);
    end
    stimval{k} = flipud(stimval{k});
end
CSR_JN_ns = bsxfun(@minus,nlens,abs(tinds));
CSR = sum(CSR_JN,3);
CSR_ns = sum(CSR_JN_ns,1);
if(nstim > 1)
    CSR_JN_ns = bsxfun(@minus,CSR_ns,CSR_JN_ns);
    CSR_JN = bsxfun(@rdivide,bsxfun(@minus,CSR,CSR_JN),permute(((CSR_JN_ns==0) + CSR_JN_ns).',[3 1 2]));
end
CSR = bsxfun(@rdivide,CSR,(CSR_ns==0) + CSR_ns);
CSRf = fft(bsxfun(@times,CSR,w),[],2);
CSR_JNf = fft(bsxfun(@times,CSR_JN,w),[],2);

% fft_AutoCrossCorr.m territory

% Take ffts (windowing moved above)
CSR_JNvf = zeros(1,nyqind);
JNv = (nstim-1)*(nstim-1)/nstim;
inds0 = 1:nyqind;
for ib=1:NBAND
    for it=1:nyqind
        CSR_JNvf(it) = JNv*complex(cov(squeeze(real(CSR_JNf(ib,it,:)))), cov(squeeze(imag(CSR_JNf(ib,it,:)))));
    end
    rmean = real(CSRf(ib,inds0));
    imean = imag(CSRf(ib,inds0));
    CSRf(ib,1:nyqind)= complex(rmean,imean);

    logics = (abs(rmean) < 0.5*sqrt(real(CSR_JNvf))) & (abs(imean) < 0.5*sqrt(imag(CSR_JNvf)));
    itstart = max(sum(find(logics,1,'first')),1);
    itend = sum(find(cumsum(logics)==3,1,'last'));
    itend(itend==0) = nyqind;
    
    inds = itstart+1:nyqind;
    expval = exp(-0.5*((inds-itstart)./(itend-itstart)).^2);
    CSRf(ib,inds) = CSRf(ib,inds).*expval;
    CSR_JNf(ib,inds,:) = bsxfun(@times,CSR_JNf(ib,inds,:),expval);
end
inds=2:nyqind;
CSRf(:,ntvec+2-inds) = conj(CSRf(:,inds));
CSR_JNf(:,ntvec+2-inds,:) = conj(CSR_JNf(:,inds,:));

% cal_Strf.m territory

% Do the matrix inversion for each frequency
CSfnorm=-inf;
CSf_full = zeros(NBAND,NBAND,nyqind);
CSftemp = zeros(NBAND);
for iff=1:nyqind
    CSftemp(tril(true(NBAND))) = conj(CSf(:,iff));
    CSf_full(:,:,iff) = CSftemp + CSftemp' - diag(diag(CSftemp));
    CSfnorm = max(CSfnorm,norm(CSf_full(:,:,iff)));
end

totlen = sum(nlens);
uncert = zeros(NBAND,ntvec,ntol);
strfJN_std = zeros(NBAND,ntvec,nstim,ntol);
ffor=zeros(NBAND,ntvec,ntol);
fforJN=zeros(NBAND,ntvec,nstim,ntol);
for iff=1:nyqind
    [u,s,v] = svd(CSf_full(:,:,iff));
    for ti = 1:ntol
        is = 1.0./diag(s);
        ranktol=tol(ti)*CSfnorm;
        ranktest = rank(CSf_full(:,:,iff),ranktol);
        inds=ranktest+1:NBAND;
        is(inds)=exp(-(inds-ranktest).^2/8)/ranktol;
        temp=v*diag(is)*u';

        ffor(:,iff,ti) = temp*CSRf(:,iff);
        for iJN = 1:nstim
            fforJN(:,iff,iJN,ti) = temp*((totlen*CSRf(:,iff)-(totlen-nlens(iJN)).*CSR_JNf(:,iff,iJN))/nlens(iJN));
        end
        % fforJN(:,iff,:,ti) = temp*bsxfun(@rdivide,bsxfun(@minus,totlen*CSRf(:,iff),bsxfun(@times,totlen-nlens.',squeeze(CSR_JNf(:,iff,:)))),nlens.');
    end
end
inds = 2:nyqind;
ffor(:,ntvec+2-inds,:) = conj(ffor(:,inds,:));
fforJN(:,ntvec+2-inds,:,:) = conj(fforJN(:,inds,:,:));

% Reconstruct full STRF and STRF_JNs, with causal window if necessary
wcausal = (atan(tinds)+pi/2)/pi;
if(~forceCausal)
    wcausal = ones(size(wcausal));
end
strf = bsxfun(@times,real(ifft(ffor,[],2)),wcausal);
strfJN = bsxfun(@times,real(ifft(fforJN,[],2)),wcausal);
for iJN=1:nstim
    strfJN_std(:,:,iJN,:) = std(strfJN(:,:,[1:(iJN-1) (iJN+1):nstim],:),0,3)/sqrt(nstim-1);
end
uncert(:,:,:) = max(strfJN_std,[],3);

predInfo = zeros(ntol,1);
cc = zeros(ntol,1);
predInfoNL = zeros(ntol,1);
ccNL = zeros(ntol,1);
psth_est = cell(nstim,2,ntol);

if(runPredVal)
    %% 2. Prediction and validation
    for ti = 1:ntol
        if(doNL)
            [cc(ti),predInfo(ti),psth_est(:,:,ti),ccNL(ti),predInfoNL(ti)] = calcPredCC(strfJN(:,:,:,ti),[-winlen winlen],stims,resps,doNL,true);
        else
            [cc(ti),predInfo(ti)] = calcPredCC(strfJN(:,:,:,ti),[-winlen winlen],stims,resps,doNL,true);
        end
    end
end
