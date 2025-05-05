function out = make(dirname,errfail)
%MAKE Make all .c files in a directory.
%   MAKE() mexes all .c files in the current directory and throws an error 
%   if any of the compiles failed.
%   
%   MAKE(DIRNAME) mexes all files in the current directory referenced by 
%   the string DIRNAME and throws an error if any of the compiles failed.
% 
%   MAKE([],ERRFAIL) mexes all .c files in the current directory and if any
%   of the compiles fail, it will throw an error if ERRFAIL = 'error', 
%   throw a warning if ERRFAIL = 'warn', or do nothing if ERRFAIL = 'none'.
% 
%   MAKE(DIRNAME,ERRFAIL) mexes all .c files in the directory DIRNAME
%   responding to compile errors based on ERRFAIL above.
%
%   OUT=MAKE(...) mexes all .c files as above and returns the number of
%   failed compiles. (NOTE: default behavior with an output is 
%   ERRFAIL='none').
%
%05/02/08 edlarson@bu.edu
% 

if(strcmp(computer,'GLNXA64') || strcmp(computer,'GLNX86'))
    msl = '/';
else
    msl = '/';
end

if(nargin < 1 || isempty(dirname))
    dirname = ['.' msl];
end
if(nargin < 2)
    if(nargout < 1)
        errfail = 'error';
    else
        errfail = 'none';
    end
end

if(~ischar(dirname))
    error('dirname must be a string');
end

if(dirname(end) ~= msl)
    dirname = [dirname msl];
end

temp = dir([dirname '*.c']);

oldFolder = cd(dirname);
result = zeros(1,length(temp));
for ii = 1:length(temp)
    result(ii) = mex(temp(ii).name);
end
cd(oldFolder);

out = sum(logical(result));
if(out)
    mystr = sprintf('Compile failed on %d of %d files:\n%s',out,length(result),sprintf('%s\n',temp(result~=0).name));
    if(strcmp(errfail,'warn'))
        warning(mystr); %#ok<SPWRN>
    elseif(~strcmp(errfail,'none'))
        error(mystr); %#ok<SPERR>
    end
end
