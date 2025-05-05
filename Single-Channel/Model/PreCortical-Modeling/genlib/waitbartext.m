function varargout = waitbartext(x, message, barlen, chars)
% WAITBARTEXT(X, MESSAGE, BARLEN, CHARS)
%   Displays a waitbar at the top of the command prompt. WAITBARTEXT uses a
%   persistent variable to erase the proper number of characters before
%   printing the progress bar. Thus, for proper operation, the progress
%   must start at 0 (i.e., for each run, the first argument to X must
%   be 0). However, if the first bar ran to completion (i.e., the last
%   value of X was 1), then it will also behave properly (in this case, by
%   starting a new bar rather than replacing the old one). MESSAGE is the
%   same as in the MATLAB function WAITBAR, and is optional or can be
%   empty. BARLEN is a scalar indicating the number of character-wide
%   increments in the bar, and is also optional or can be empty. It
%   defaults to 50. CHARS is a 2-element string that indicates what
%   characters to use for the completed and uncompleted portions of the
%   bar. It too is optional and can be designated as the empty matrix (the
%   default is '+-').

persistent numChars startTime;
if isempty(numChars) || x == 0
    numChars = 0;
	startTime = tic;
end

if x < 0 || x > 1
    warning('X must be between 0 and 1 inclusive.')
	x = 1;
end

if ~exist('message', 'var') || isempty(message) || (~ischar(message) && message == -1 && (x == 0 || x == 1))
	message = sprintf('%5.1f%% (%s)', x*100, timestring(toc(startTime)));
elseif message == -1
	message = sprintf('%5.1f%% (%s)', x*100, timestring((1/x - 1)*toc(startTime)));
end

if ~exist('barlen', 'var') || isempty(barlen)
	barlen = 20;
end

if ~exist('chars', 'var') || isempty(chars)
	chars = '+-';
end

done = floor(barlen*x);

clearLine = repmat(sprintf('\b'), 1, numChars);
outString = sprintf('|%s%s| %s\n', repmat(chars(1), 1, done), repmat(chars(2), 1, barlen - done), message);
fprintf('%s%s', clearLine, outString)

if x < 1
    numChars = length(outString);
else
    numChars = 0;
end

if nargout == 1
	varargout{1} = length(outString);
end