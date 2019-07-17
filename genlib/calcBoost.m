function [strf,uncert,strfJN,cc,predInfo,psth_est,ccNL,predInfoNL] = calcBoost(stims,resps,winlen,tol)

doNL = false;
if(nargout > 5)
    doNL = true;
end
if(numel(winlen)==1)
    winlen = [-winlen winlen];
elseif(numel(winlen)~=2)
    error('winlen must be a 1 or 2-element vector');
end

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

tinds = winlen(1):winlen(2);
ntvec = length(tinds);

strf = zeros(NBAND,ntvec,ntol);
strfJN = zeros(NBAND,ntvec,nstim,ntol);
predInfo = zeros(ntol,1);
cc = zeros(ntol,1);
predInfoNL = zeros(ntol,1);
ccNL = zeros(ntol,1);
psth_est = cell(nstim,2,ntol);

% Get the stimulus variability compared to the response variability
stemp = vertcat(stims{:});
svar = 0;
for bi = 1:NBAND
    svar = svar + var(stemp(:,bi));
end
svar = svar/NBAND;
clear stemp;
r = cell(nstim,1);
for si = 1:nstim
    r{si} = single(mean(resps{si},2));
end
e = single(sqrt(var(vertcat(r{:}))/svar)/250);
%stimConst = single(std(vertcat(r{:}))/50);
rJN = vertcat(r{:}); clear r;

tshifts = -winlen(1):-1:-winlen(2);
SJNe = zeros(sum(nlens),NBAND*ntvec,'single');
% S = cell(size(stims));
for si = 1:nstim
    % S{si} = zeros(nlens(si),NBAND*ntvec,'single');
    for ti = 1:ntvec
        foffs = (ti-1)*NBAND+(1:NBAND);
        inds1 = (max(-tshifts(ti),0)+1) : (nlens(si)-max(tshifts(ti),0));
        inds2 = (max(tshifts(ti),0)+1)  : (nlens(si)-max(-tshifts(ti),0));
        SJNe(inds1+sum(nlens(1:si-1)),foffs) = stims{si}(inds2,:)*e;
        % S{si}(inds1,foffs) = stims{si}(inds2,:);
    end
    % SJNe(:,NBAND*ntvec) = stimConst;
end

% rJN = cell(nstim,1);
% SJNe = cell(nstim,1);
% rmSh = cell(nstim,1);
% for si = 1:nstim
%     inds = [1:si-1 si+1:nstim];
%     rJN{si} = vertcat(r{si});
%     SJNe{si} = vertcat(S{si})*e;
%     rmSh{si} = rJN{si}; % Residual error is initially just the response (start 0 strf)
% end
% h = zeros(NBAND*ntvec,nstim);
% SJNe = vertcat(S{:})*e; clear S;
h = zeros(NBAND*ntvec,1);
rmSh = rJN; % Residual error is initially just the response (start 0 strf)

lastImproved = true;
improveNeeded = 0;
ii = 1;
prevPred = -2*eps; %#ok<NASGU>
thisPred = -eps;
sgns = zeros(tol(end),1);
bestInds = zeros(tol(end),1);
offset = zeros(ntol,1);
while(ii <= tol(end) && lastImproved)
    % Calculate the MSE between residual response and changes, update STRF and residual error
    % for si = 1:nstim
    %     % Calculate the MSE between residual response and changes, update STRF and residual error
    %     [bestInds(ii,si),sgns(ii,si)] = calcBoostHelper(SJNe{si},rmSh{si},feature('numCores'));
    %     h(bestInds(ii,si),si) = h(bestInds(ii,si),si) + e*sgns(si,ii);
    %     rmSh{si} = rmSh{si} - sgn*SJNe{si}(:,bestInds(ii,si));
    % end
    [bestInds(ii),sgns(ii)] = calcBoostHelper(SJNe,rmSh,feature('numCores'));
    %if(bestInds(ii) <= NBAND*ntvec)
    h(bestInds(ii),1) = h(bestInds(ii),1) + e*sgns(ii);
    rmSh = rmSh - sgns(ii)*SJNe(:,bestInds(ii));
    %else
    %    h(bestInds(ii),1) = h(bestInds(ii),1) + stimConst*sgns(ii);
    %    rmSh = rmSh - sgns(ii)*stimConst;
    %end

    % Save results, perhaps
    improveNeeded = improveNeeded+1;
    ind = find(ii==tol,1);
    if(~isempty(ind))
        for si = 1:nstim
            % strfJN(:,:,si,ind) = reshape(h(:,si),NBAND,ntvec);
            strfJN(:,:,si,ind) = reshape(h(1:NBAND*ntvec,1),NBAND,ntvec);
        end
        strf(:,:,ind) = mean(strfJN(:,:,:,ind),3);
        if(doNL)
            [cc(ind),predInfo(ind),psth_est(:,:,ind),ccNL(ind),predInfoNL(ind)] = calcPredCC(strfJN(:,:,:,ind),winlen,stims,resps,doNL,false,offset(ind));
        else
            [cc(ind),predInfo(ind)] = calcPredCC(strfJN(:,:,:,ind),winlen,stims,resps,doNL,false);
        end
        prevPred = thisPred;
        thisPred = predInfo(ind);

        % Make sure we make at least a 0.1% improvement per iteration
        lastImproved = (thisPred-prevPred)/max(thisPred,eps) > (0.0005)*improveNeeded;
        if(~lastImproved) % Make it so you keep the previous version
            predInfo(ind) = -predInfo(ind);
        end
        improveNeeded = 0;
    end
    ii = ii+1;
end

strfJN_std = zeros(NBAND,ntvec,nstim,ntol);
% for iJN=1:nstim % Will all necessarily be zeros if not jackknifing
%     strfJN_std(:,:,iJN,:) = std(strfJN(:,:,[1:(iJN-1) (iJN+1):nstim],:),0,3)/sqrt(nstim-1);
% end
uncert = max(strfJN_std,[],3);

