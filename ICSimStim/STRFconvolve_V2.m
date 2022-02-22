function [spkcnt_on,spkcnt_off,onset_rate,offset_rate,spktimes_on,spktimes_off]=...
    STRFconvolve_V2(strf,stim_spec,mean_rate,trialn,songn,t_ref)
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
% frate(find(frate<0))=zeros(size(find(frate<0))); % half-wave rec

% offset rate
offset_rate = -frate + max(frate)*0.85; %-frate + max(frate)*0.6;
firstneg = find(offset_rate < 0,1,'first');
offset_rate(1251:firstneg-1) = 0;
offset_rate(offset_rate < 0) = 0;

% onset rate
onset_rate = frate;
onset_rate(onset_rate < 0) = 0;

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
spktimes_on=cell(trialn,1);
spkcnt_on=0;

spktimes_off=cell(trialn,1);
spkcnt_off=0;

% use rand seed
% load('rand_seed142307.mat')
rng('shuffle')

for i=1:trialn
    if exist('rand_seed','var')
        % disp('using seed random numbers')
        [~,temptspk_on]=spike_generator_seed_rr(onset_rate,t,rand_seed(:,i,songn),t_ref);
        spktimes_on{i,1}=temptspk_on*1000;
        
        [~,temptspk_off]=spike_generator_seed_rr(offset_rate,t,rand_seed(:,i,songn),t_ref);
        spktimes_off{i,1}=temptspk_off*1000;
    else
        % disp('using novel random numbers')
        [~,temptspk_on]=spike_generator_rr(onset_rate,t,t_ref);
        spktimes_on{i,1}=temptspk_on*1000;
        
        [~,temptspk_off]=spike_generator_rr(offset_rate,t,t_ref);
        spktimes_off{i,1}=temptspk_off*1000;
    end
    spkcnt_on=spkcnt_on+length(temptspk_on);
    spkcnt_off=spkcnt_off+length(temptspk_off);
end
if trialn==1
    spktimes_on=cell2mat(spktimes_on);
    spktimes_off=cell2mat(spktimes_off);
end
spkcnt_on=spkcnt_on/trialn;
spkcnt_off=spkcnt_off/trialn;


