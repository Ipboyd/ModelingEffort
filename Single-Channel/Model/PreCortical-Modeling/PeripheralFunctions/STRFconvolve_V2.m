% function [spkcnt_on,onset_rate,spktimes_on]=...
%     STRFconvolve_V2(strf,stim_spec,mean_rate,trialn,songn,t_ref,t_ref_rel,rec)
 
function [onset_rate,offset_rate]=...
     STRFconvolve_V2(strf,stim_spec,mean_rate)
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
elseif nargin<3||nargin>9
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
frate(isnan(frate)) = 0;
% frate(find(frate<0))=zeros(size(find(frate<0))); % half-wave rec

% offset rate
offset_rate = -frate + max(frate); %-frate + max(frate)*0.6;
firstneg = find(offset_rate <= 0,1,'first');

if firstneg > 5500, firstneg = 2501; end % for AM stimuli
%firstneg = find(offset_rate <= 0,1,'first');

offset_rate(1:firstneg-1) = 0;
offset_rate(offset_rate < 0) = 0;

% onset rate
onset_rate = frate;
onset_rate(onset_rate < 0) = 0;

% cosine ramp to onset
% onset_rate(2501:4500) = onset_rate(2501:4500).*cos(pi/4000 .* (0:1999) - pi/2)';

% frate=abs(frate); % full-wav rec
% m=mean(power.^(frate(isfinite(frate(:, 1)), :)));
% normalize spike rate
% if power==1
%     frate=frate*mean_rate/mean(frate(isfinite(frate(:, 1)), :));
% else
%     frate=power.^(frate)*mean_rate;
% %         frate=power.^(frate)*mean_rate/m;
% end



