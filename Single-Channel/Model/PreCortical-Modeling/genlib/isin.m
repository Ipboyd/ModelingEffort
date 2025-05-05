%function OUT = isin(WHICH,INWHAT) determines which elements in WHICH 
%   appear in INWHAT. Both WHICH and INWHAT must be doubles, and are 
%   treated as vectors. The output OUT is a logical vector of size 
%   [numel(WHICH) 1] saying which elements actually appear in INWHAT. This 
%   function is equivalent to calling
%   
%     OUT = any(bsxfun(@eq,WHICH(:),INWHAT(:).'),2);
%   
%   but it should be faster and require less memory. It /should/ also be 
%   faster than pulling out indices using intersect().
%   