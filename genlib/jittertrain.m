% function out = jittertrain(x,sig)
%   OUT = JITTERTRAIN(X,SIG) jitters the spike train vector X by adding 
%   white Gaussian noise with standard deviation SIG and mean 0, rounding 
%   the result, and taking the unique elements.
%

% old code that has been improved and MEX'ed:
%
% spikeinds = find(x==1);
% newtrain = histc(min(length(x), max(1, spikeinds+round(sig*randn(size(spikeinds))))),1:length(x));
% extras = find(newtrain > 1)';
% if(~isempty(extras))
%     for ii = extras % Take care of duplicates
%         for di = 2:newtrain(ii) % For each duplicate...
%             emptinds = find(newtrain == 0);
%             [junk, ind] = min(abs(ii-emptinds));
%             newtrain(emptinds(ind)) = 1;
%         end
%         newtrain(ii) = 1;
%     end
% end
% 
% out = newtrain;
