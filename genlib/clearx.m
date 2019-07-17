function clearx(varargin)
%CLEARX  Clear unlisted variables.
%   CLEARX('VAR1', 'VAR2', ...) clears all the variables in
%   the caller workspace EXCEPT for those variables whose
%   names are given as arguments.
%
%   CLEARX(VARS) does the same where VARS is a cell array of
%   strings that are the names of the variables to keep.
%
%   See also clear, clearvars, who, whos, mlock, munlock, persistent.

% Written by Ross Maddox, rkmaddox@bu.edu, 2010/05/17

if nargin == 1
	toKeep = varargin{1};
	if ischar(toKeep)
		toKeep = {toKeep};
	end
else
	toKeep = varargin;
end
wsVars = evalin('caller', 'who');
for ii = 1:length(toKeep)
	for jj = length(wsVars):-1:1
		if strcmp(wsVars{jj}, toKeep{ii})
			wsVars(jj) = [];
		end
	end
end
wsVars{end+1} = char('A' - 1 + randi(26, [1, 60])); % create a very unlikely variable name
assignin('caller', wsVars{end}, wsVars)
evalin('caller', sprintf('clear(%s{:})', wsVars{end}))