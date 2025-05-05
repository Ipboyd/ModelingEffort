function x = delta(t)
% X = DELTA(T)
% Kroneker delta function.  X = 0 everywhere except at T = 0.
x = t == 0;