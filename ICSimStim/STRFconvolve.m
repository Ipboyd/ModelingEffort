function [spkcnt,frate,spike_times]=STRFconvolve(strf,stim_spec,mean_rate,trialn,songn,t_ref)
% 
%
%% Inputs
% strf
% stim_spec
%% Output
% frate:    firing rate as a function of time
% spike_times
% 20190905 removed rand_seed from the input

if nargin==3
    trialn=1;
    songn=[];
    t_ref=3;
elseif nargin==4
    songn=[];
    t_ref=3;
elseif nargin==5
    t_ref=3;
elseif nargin<3||nargin>6
    disp('wrong number of inputs')
    return
end
t=strf.t;
%% convolve STRF and stim
% Initialize strflab global variables with our stim and responses
strfData(stim_spec, zeros(size(stim_spec)));
%
[~, frate] = strfFwd_Junzi(strf);
% frate=abs(frate);
frate=frate*mean_rate;
frate(find(frate<0))=zeros(size(find(frate<0))); % half-wave rec
% frate=abs(frate); % full-wav rec
% m=mean(power.^(frate(isfinite(frate(:, 1)), :)));
% normalize spike rate
% if power==1
%     frate=frate*mean_rate/mean(frate(isfinite(frate(:, 1)), :));
% else
%     frate=power.^(frate)*mean_rate;
% %         frate=power.^(frate)*mean_rate/m;
% end
%% generate spikes
spike_times=cell(trialn,1);
spkcnt=0;

% use rand seed
% load('rand_seed142307.mat')
rng('shuffle')

for i=1:trialn
    if exist('rand_seed','var')
        % disp('using seed random numbers')
        [~,temptspk]=spike_generator_seed_rr(frate,t,rand_seed(:,i,songn),t_ref);
        spike_times{i,1}=temptspk*1000;
    else
        % disp('using novel random numbers')
        [~,temptspk]=spike_generator_rr(frate,t,t_ref);
        spike_times{i,1}=temptspk*1000;
    end
    spkcnt=spkcnt+length(temptspk);
end
if trialn==1
    spike_times=cell2mat(spike_times);
end
spkcnt=spkcnt/trialn;


