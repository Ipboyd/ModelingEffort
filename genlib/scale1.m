function a = scale1(a)
%SCALE1(a)
% Scales an n-dimensional real vector to within [-1,+1].

a = a/max(abs(a(:)));