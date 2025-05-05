% function out = addspikes(x,f,fs)
%   OUT = ADDSPIKES(X,F,FS) adds or removes poisson spikes at a rate 
%   specified by F for the spike train X at sample rate FS.
%

% old code that has been improved and MEX'ed:
%
% % Generate some poisson spikes
% ds = fs/abs(f);
% news = find(rand(size(x)) <= 1/ds);
% 
% % Either add or remove those spikes
% if(f >= 0)
%     x(news) = x(news) + 1;
%     extras = find(x > 1)';
%     if(~isempty(extras))
%         for ii = extras % Take care of duplicates
%             for di = 2:x(ii) % For each duplicate...
%                 emptinds = find(x == 0);
%                 [junk, ind] = min(abs(ii-emptinds));
%                 x(emptinds(ind)) = 1;
%             end
%             x(ii) = 1;
%         end
%     end
% else
%     for ii = 1:length(news)
%         temp = find(x) - news(ii);
%         [dx,di] = min(abs(temp));
%         if(dx < ds)
%             x(news(ii) + sign(temp(di))*dx) = 0;
%         end
%     end
% end
