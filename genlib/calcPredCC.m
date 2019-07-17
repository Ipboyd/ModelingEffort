% A helper function for calcSTRF and calcBoost
% function [cc,predInfo,psth_est,ccNL,predInfoNL] = calcPredCC(strfJN,winlen,stims,resps,doNL,removeMean)

function [cc,predInfo,psth_est,ccNL,predInfoNL] = calcPredCC(strfJN,winlen,stims,resps,doNL,removeMean,offset)
if(nargin < 6 || isempty(removeMean))
    removeMean = true;
end
if(nargout > 2)
    doNL = true;
end
if(nargin < 7 || isempty(offset))
    offset = 0;
end
[NBAND,junk,nstim] = size(strfJN); %#ok<ASGLU>
psth_est = cell(nstim,2);
ntrialss = cellfun(@size,resps,repmat({2},nstim,1));
nlens = cellfun(@size,stims,repmat({1},nstim,1));
a=cellfun(@sum,stims,repmat({1},nstim,1),'UniformOutput',false);
stim_avg = sum(bsxfun(@times,vertcat(a{:}),ntrialss),1)/(nlens.'*ntrialss);
psth_avg = mean(vector(vertcat(resps{:})));

%% 2. Prediction
twin = max(abs(winlen));
strfJN2 = zeros(NBAND,2*twin+1,nstim);
strfJN2(:,winlen(1)+twin+1:winlen(2)+twin+1,:) = strfJN;
padlen = 2*(1+twin+twin^2);
pad = zeros(padlen,1);
template = (-twin:twin)*2*(twin+1) -twin + padlen;  %  This is how the matrix adds up to get what we want.
ctinds = bsxfun(@plus,(1:max(nlens))*(2*twin+1),template.');
avg_spike1 = cell(nstim,1);
avg_spike2 = cell(nstim,1);
est_spike = cell(nstim,1);
est_spike_nl = cell(nstim,1);
stimval = cell(nstim,1);
for k = 1:nstim
    if(~removeMean)
        stimval{k} = stims{k};
    else
        stimval{k} = bsxfun(@minus,stims{k},stim_avg);
    end

    % Compute the PSTHs
    avg_spike1{k} = mean(resps{k}(:,2:2:ntrialss(k)),2);
    avg_spike2{k} = mean(resps{k}(:,1:2:ntrialss(k)),2);
    if ntrialss(k) == 1
        avg_spike1{k} = avg_spike2{k};
    end

    % Compute the filtered JN STRF
    index_to_use = 1 + mod((1:(nstim-2))+k, nstim);
    c_t = [pad; vector(fliplr(mean(strfJN2(:,:,index_to_use),3))'*stimval{k}.') ; pad];
    est_spike{k} = max(sum(c_t(ctinds(:,1:nlens(k)))).',0) + offset;
    if(removeMean)
        est_spike{k} = est_spike{k} + psth_avg;
    end

    if(doNL)
        c_t = [pad; vector(fliplr(mean(strfJN2,3))'*stims{k}.') ; pad];
        psth_est{k,1} = mean(resps{k},2);
        if(~removeMean)
            psth_est{k,2} = sum(c_t(ctinds(:,1:nlens(k)))).' + offset; % No nonlinear > 0 here
        else
            psth_est{k,2} = sum(c_t(ctinds(:,1:nlens(k)))).' + psth_avg + offset; % No nonlinear > 0 here
        end
    end
end

% Add a(n optimal?) nonlinearity
if(doNL)
    ord = 5;
    p = polyfit(vertcat(psth_est{:,2}),vertcat(psth_est{:,1}),ord).';
    for k = 1:nstim
        psth_est{k,3} = bsxfun(@power,psth_est{k,2},(ord:-1:0))*p;
        est_spike_nl{k} = bsxfun(@power,est_spike{k},(ord:-1:0))*p;
    end
end

%% 3. Validation
spike_est1 = vertcat(avg_spike1{:});
spike_est2 = vertcat(avg_spike2{:});
spike_pre = vertcat(est_spike{:});
if(doNL)
    spike_pre_nl = vertcat(est_spike_nl{:});
end

% Calculate predicted info
ntrials_proper = min(ntrialss);
temp = [length(spike_pre) length(spike_est1) length(spike_est2)];
the_min = min(temp);
if range(temp) == 1  %If there's a 1 ms fault, it can be because of even/odd window sizes, and we don't care.
    spike_pre = spike_pre(1:the_min);
    if(doNL)
        spike_pre_nl = spike_pre_nl(1:the_min);
    end
    spike_est1 = spike_est1(1:the_min);
    spike_est2 = spike_est2(1:the_min);
end
% Calculate CC and predInfo
cc = calc_cc_ratio(spike_est1, spike_est2, spike_pre, [5 8 29], ntrials_proper);
predInfo = SNRinfo_no_cutoff(ntrials_proper, [spike_pre-mean(spike_pre) (spike_est1+spike_est2)/2-psth_avg spike_est1-psth_avg spike_est2-psth_avg], 128);
if(doNL)
    ccNL = max(calc_cc_ratio(spike_est1, spike_est2, spike_pre_nl, [5 8 29], ntrials_proper));
    predInfoNL = SNRinfo_no_cutoff(ntrials_proper, [spike_pre_nl-mean(spike_pre_nl) (spike_est1+spike_est2)/2-psth_avg spike_est1-psth_avg spike_est2-psth_avg], 128);
else
    ccNL = 0;
    predInfoNL = 0;
end


    
function cc_ratio_max = calc_cc_ratio(spike_est1, spike_est2, spike_pre, widthvector, ntrials_proper)

% Smoothing parameters when calculating CC
[cc_two_halves_corrval1, cc_two_halves_corrval1_low, cc_two_halves_corrval1_high] = cc_plot_est(spike_est1, spike_est2, widthvector, 1);

notzeroind = find(cc_two_halves_corrval1 ~= 0);
temp = zeros(size(cc_two_halves_corrval1));
temp(notzeroind) =(-ntrials_proper+ntrials_proper*sqrt(1./cc_two_halves_corrval1(notzeroind)))/2;
cc_two_halves_corrval = zeros(size(cc_two_halves_corrval1));
cc_two_halves_corrval(notzeroind)=1./(temp(notzeroind)+1);

clear notzeroind temp
notzeroind = find(cc_two_halves_corrval1_high ~= 0);
temp = zeros(size(cc_two_halves_corrval1_high));
temp(notzeroind) =(-ntrials_proper+ntrials_proper*sqrt(1./cc_two_halves_corrval1_high(notzeroind)))/2;
cc_two_halves_corrval_high = zeros(size(cc_two_halves_corrval1_high));
cc_two_halves_corrval_high(notzeroind)=1./(temp(notzeroind)+1);

% Take the square root to get r values and not r square - following
% normalization of equation 8 in Hsu et al.
cc_two_halves_corrval=sqrt(cc_two_halves_corrval);
cc_two_halves_corrval_high=sqrt(cc_two_halves_corrval_high);

[cc_spike_pre_corrval1, cc_spike_pre_corrval1_low, cc_spike_pre_corrval1_high] = cc_plot_est((spike_est1+spike_est2)/2,spike_pre,widthvector, 0);

clear notzeroind
notzeroind = find(cc_two_halves_corrval1 ~=0 );
cc_spike_pre_corrval = zeros(size(cc_spike_pre_corrval1));
cc_spike_pre_corrval(notzeroind)=cc_spike_pre_corrval1(notzeroind).*(1+sqrt(1./cc_two_halves_corrval1(notzeroind)))./(-ntrials_proper+ntrials_proper*sqrt(1./cc_two_halves_corrval1(notzeroind))+2);

clear notzeroind
notzeroind = find(cc_two_halves_corrval1_low ~=0 );
cc_spike_pre_corrval_low = zeros(size(cc_spike_pre_corrval1_low));
cc_spike_pre_corrval_low(notzeroind)=cc_spike_pre_corrval1_low(notzeroind).*(1+sqrt(1./cc_two_halves_corrval1_low(notzeroind)))./(-ntrials_proper+ntrials_proper*sqrt(1./cc_two_halves_corrval1_low(notzeroind))+2);

cc_spike_pre_corrval=sqrt(cc_spike_pre_corrval);
cc_spike_pre_corrval_low=sqrt(cc_spike_pre_corrval_low);

% Find the best ratio of predicted_CC and CC
cc_ratio = zeros(size(cc_spike_pre_corrval));
notzeroind = find(cc_two_halves_corrval~=0);
cc_ratio(notzeroind) = cc_spike_pre_corrval(notzeroind)./cc_two_halves_corrval(notzeroind);
for icc=1:length(notzeroind)
   if (cc_ratio(icc) > 1.0) 
       if (cc_spike_pre_corrval_low(icc) < cc_two_halves_corrval_high(icc) )
           cc_ratio(icc) = 1.0;
       end
   end
end
cc_ratio_max = max(cc_ratio);

function info = SNRinfo_no_cutoff(ntrials, x, nFFT)

[fpxy, cxyo, cxyo_u, cxyo_l]=mtchd_JN(x(:,1:2),nFFT);
cxy_notnormalized=cxyo(:,1,2);
cxy_notnormalizedlo=cxyo_l(:,1,2);
[fpxy, cxyo, cxyo_u, cxyo_l]=mtchd_JN(x(:,3:4),nFFT);
cxy_notnormalizedpsth=cxyo(:,1,2);
cxy_notnormalizedpsthup=cxyo_u(:,1,2);
cxy_notnormalizedpsthlo=cxyo_l(:,1,2);
cxy_notnormalizedpsth=cxy_notnormalizedpsth.^2;
cxy_notnormalizedpsthup=cxy_notnormalizedpsthup.^2;
cxy_notnormalizedpsthlo=cxy_notnormalizedpsthlo.^2;
cxy_notnormalized=cxy_notnormalized.^2;
cxy_notnormalizedlo=cxy_notnormalizedlo.^2;
cxydown=cxy_notnormalizedlo;
cxy=cxy_notnormalized;

%changes coherences to that of one trial (normalization) except for where cxy=0;
cxypsthup=cxy_notnormalizedpsthup;
cxypsth=cxy_notnormalizedpsth;

index=find(cxypsth~=0);
k=(-ntrials+ntrials*sqrt(1./cxypsth(index)))/2;
cxypsthnew=1./(k+1);
cxypsth(index)=cxypsthnew;
clear cxypsthnew

index=find(cxypsthup~=0);
kup=(-ntrials+ntrials*sqrt(1./cxypsthup(index)))/2;
cxypsthnewup=1./(kup+1);
cxypsthup(index)=cxypsthnewup;
clear cxypsthnewup

%now cxypsth's are the normalized coherence of one spike train with the actual meanrate.
cxydown(index)=cxy_notnormalizedlo(index).*(1+sqrt(1./cxy_notnormalizedpsthlo(index)))./(-ntrials+ntrials*sqrt(1./cxy_notnormalizedpsthlo(index))+2);
cxy(index)=cxy_notnormalized(index).*(1+sqrt(1./cxy_notnormalizedpsth(index)))./(-ntrials+ntrials*sqrt(1./cxy_notnormalizedpsth(index))+2);
infodown=-fpxy(2)*sum(log2(1-cxydown));
info=-fpxy(2)*sum(log2(1-cxy));
infopsthup=-fpxy(2)*sum(log2(1-cxypsthup));
infopsth=-fpxy(2)*sum(log2(1-cxypsth));

if infopsth < info
    if infodown <= infopsthup
        info = infopsth;
    end
end

function [corrval, corrval_low, corrval_high] = cc_plot_est(spike1, spike2, widthvector, smoothflag)

i = 1;
for width=widthvector(1):widthvector(2):widthvector(3)
    window_width = width;  
    wind1 = hanning(window_width)./sum(hanning(window_width));
    sest = conv(spike1,wind1);

    if ( smoothflag )  % Smooth both signals
        spre = conv(spike2,wind1);
    else
        if mod(window_width,2)==0
            window_width=window_width+1; 
            wind2 = zeros(window_width,1);
            wind2((window_width+1)/2) = 1;
            wind2 = wind2(1:end-1);
        else
            wind2 = zeros(window_width,1);
            wind2((window_width+1)/2) = 1;
        end
        spre = conv(spike2,wind2);
    end
    [rcenter rp rlow rhigh] = corrcoef(spre,sest);
    corrval(i)= max(diag(rcenter,1),0);
    corrval_low(i) = diag(rlow,1);
    corrval_high(i) = diag(rhigh,1);
    corrval_low(i) = max(0.0, corrval_low(i));
    corrval_high(i) = max(0.0, corrval_high(i));
    i = i+1;
end
corrval=corrval.^2;
corrval_low = corrval_low.^2;
corrval_high = corrval_high.^2;

function [fo, meanP, Pupper, Plower]=mtchd_JN(x, nFFT)

WinLength = nFFT;
nOverlap = WinLength/2;
NW = 3;
Detrend = '';
nTapers = 2*NW -1;

winstep = WinLength - nOverlap;
nChannels = size(x, 2);
nSamples = size(x,1);

if nSamples == 1 
	x = x';
	nSamples = size(x,1);
	nChannels = 1;
end;
nFFTChunks = round(((nSamples-WinLength)/winstep));

varP = zeros(nFFT, nChannels, nChannels);
[Tapers V]=dpss(WinLength,NW,nTapers, 'calc');
Periodogram = complex(zeros(nFFT, nTapers, nChannels)); % intermediate FFTs
JN = complex(zeros(nFFT, nChannels, nChannels, nFFTChunks));  %jackknifed csd
y=complex(zeros(nFFT, nChannels, nChannels)); % output array for csd
Py=zeros(nFFT, nChannels, nChannels); % output array for psd's

TaperingArray = repmat(Tapers, [1 1 nChannels]);
for j=1:nFFTChunks
	Segment = x((j-1)*winstep+[1:WinLength], :);
	if (~isempty(Detrend))
		Segment = detrend(Segment, Detrend);
	end;
	TaperedSegments = bsxfun(@times, TaperingArray, permute(Segment,[1 3 2]));
	Periodogram(:,:,:) = fft(TaperedSegments,nFFT);

	% Now make cross-products of them to fill cross-spectrum matrix
	for Ch1 = 1:nChannels
		for Ch2 = Ch1:nChannels % don't compute cross-spectra twice
			Temp3 = Periodogram(:,:,Ch1) .* conj(Periodogram(:,:,Ch2));

            %eJ and eJ2 are the sum over all the tapers.
            JN(:,Ch1, Ch2, j)=sum(Temp3, 2)/nTapers;
		end
	end
end
y = sum(JN,4);

% now fill other half of matrix with complex conjugate
for Ch1 = 1:nChannels
	for Ch2 = (Ch1+1):nChannels % don't compute cross-spectra twice
		y(:, Ch2, Ch1) = y(:,Ch1,Ch2);
        Py(:, Ch1, Ch2) = atanh(abs(y(:,Ch1,Ch2)./sqrt(abs(y(:,Ch1,Ch1)).*abs(y(:,Ch2,Ch2)))));
	end
end

absJN = zeros(size(JN,1),size(JN,2));
JN = abs(bsxfun(@minus,y,JN));
for j = 1:nFFTChunks
    tempJN = JN(:,:,:,j);
    for Ch1 = 1:nChannels
        absJN(:,Ch1) = abs(tempJN(:,Ch1,Ch1));
    end
    for Ch1 = 1:nChannels
        for Ch2 = (Ch1+1):nChannels  
            tempJN(:, Ch1, Ch2) = atanh(real(tempJN(:,Ch1,Ch2)) ./ sqrt( abs(tempJN(:,Ch1,Ch1)) .* abs(tempJN(:,Ch2,Ch2)) ));
            tempJN(:, Ch1, Ch2) = nFFTChunks*Py(:, Ch1, Ch2) - (nFFTChunks-1)*tempJN(:, Ch1,Ch2);
        end
    end  
    JN(:,:,:,j) = tempJN;
end

meanP=squeeze(mean(JN,4));
for Ch1=1:nChannels
    for Ch2=Ch1:nChannels
        varP(:,Ch1, Ch2) = (1/nFFTChunks)*var(JN(:,Ch1, Ch2,:),1,4);
    end
end
stP=sqrt(varP);

Pupper = meanP + 2*stP;
Plower = meanP - 2*stP;
meanP = tanh(meanP);
Pupper = tanh(Pupper);
Plower = tanh(Plower);

if ~any(any(imag(x)))    % x purely real
	if rem(nFFT,2),    % nfft odd
		select = [1:(nFFT+1)/2];
	else
		select = [1:nFFT/2+1];
	end
    meanP = meanP(select,:,:);
    Pupper = Pupper(select,:,:);
    Plower = Plower(select,:,:);
else
	select = 1:nFFT;
end
fo = (select - 1)'/nFFT;

