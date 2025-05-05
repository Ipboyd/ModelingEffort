function out = lnspace(a,b,n)
out = a*(b/a).^[(0:n-2)/(n-1) 1];