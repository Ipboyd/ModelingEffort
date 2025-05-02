function [nx,ny] = functionalize(x,y)
% function [x,y] = functionalize(x,y)
%   [X,Y] = FUNCTIONALIZE(X,Y) turns the line with pairs (X,Y) where X and
%   Y are monotonically nondecreasing, and turns it into a function (that
%   is, for every value x in X, there is exactly one y corresponding to
%   it). This is useful for reducing staircase-type lines to functions.
%   
%   For an example, run:
% 
%     b = [1 2 2 2 3 4 5 6 7 8 8 8 8 8 9 10; 0 0 1 2 2 2 2 2 2 2 3 4 5 6 6 6];
%     [bx,by] = functionalize(b(1,:),b(2,:));
%     plot(b(1,:),b(2,:),'r',bx,by,'g');
% 
%   Modified 06/18/2008 Eric Larson & Ross Maddox
%   

nonreps = diff(x) ~= 0;
nx = x(nonreps);
ny = y(nonreps);
