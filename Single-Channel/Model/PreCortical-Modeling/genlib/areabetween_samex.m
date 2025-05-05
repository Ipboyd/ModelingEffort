function area = areabetween_samex(x, yt, y)
% function out = areabetween_samex(x,yt,y)
%   AREA = AREABETWEEN_SAMEX(...) calculates the area between two functions
%   (i.e. for every Xi there is exactly one Yi) defined by the sorted 
%   pairings (X,YT) and (X,Y). If the funtion YT is a constant, then you 
%   can also call AREABETWEEN(X1,YT,Y) where YT is a the scalar constant.
% 
%   See also: areabetween(x1,x2,y1,y2)
%
% 2008/08/07 Eric Larson edlarson@bu.edu
% 

n = size(y, 2);
if numel(yt) == 1
	yt = yt*ones(size(x));
end

if n == numel(y)
	area = areabetween(x,[],yt,y);
else
	area = zeros(n, 1);
	for ii = 1:n
		area(ii) = areabetween(x,[],yt,y(:,ii));
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % The AREABETWEEN_SAMEX below had problems with:
% %    areabetween_samex([1 2],[0 1],[1 0]) != 
% %    areabetween_samex([2 3],[0 1],[1 0])
% % Now there is a more general function areabetween(x1,x2,y1,y2), so it is
% % wrapped instead. (See above wrapping call.)
% 
% x = x(:);
% if numel(yt) == 1
% 	yt = repmat(yt, length(x), 1);
% end
% if sum(size(y) == 1)
% 	y = y(:);
% end
% numcurves = size(y, 2);
% numpoints = length(x);
% xdiff = diff(x);
% if sum(size(yt) == 1)
% 	h = y - repmat(yt(:), 1, numcurves);
% else
% 	h = y - yt;
% end
% area = zeros(size(y, 2), 1);
% for ii = 1:numcurves
% 	for jj = 1:numpoints - 1
% 		if (h(jj,ii)*h(jj + 1,ii) >= 0)
% 			area(ii) = area(ii) + abs((h(jj,ii) + h(jj + 1,ii))*xdiff(jj)/2);
% 		else
% 			xc = h(jj,ii)*xdiff(jj)/(h(jj,ii) - h(jj + 1,ii));
% 			area(ii) = area(ii) + (abs(h(jj,ii)*(xc - x(jj))) + abs(h(jj + 1,ii)*(x(jj + 1) - xc)))/2;
% 		end
% 	end
% end
