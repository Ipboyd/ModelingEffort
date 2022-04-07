function [onset_rate,offset_rate]=...
    STRFconvolve_Current(strf,stim_spec,mean_rate,offsetFrac)
%
%% Inputs
% strf
% stim_spec
%% Output
% frate:    firing rate as a function of time
% spike_times
% 20190905 removed rand_seed from the input

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
offset_rate = -frate + max(frate)*offsetFrac;
firstneg = find(offset_rate <= 0,1,'first');
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

