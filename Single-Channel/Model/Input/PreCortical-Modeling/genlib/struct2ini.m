function struct2ini(filename,Structure)
%==========================================================================
% Author:      Dirk Lohse ( dirklohse@web.de )
% Version:     0.1a
% Last change: 2008-11-13
%==========================================================================
%
% struct2ini converts a given structure into an ini-file.
% It's the opposite to Andriy Nych's ini2struct. Only
% creating an ini-file is implemented. To modify an existing
% file load it with ini2struct.m from:
%       Andriy Nych ( nych.andriy@gmail.com )
% change the structure and write it with struct2ini.
%

% Open file, or create new file, for writing
% discard existing contents, if any.
fid = fopen(filename,'w');

Sections = fieldnames(Structure);                     % returns the Sections

for i=1:length(Sections)
    Section = char(Sections(i));                       % convert to character
    
    fprintf(fid,'\n[%s]\n',Section);                       % output [Section]
    
    member_struct = Structure.(Section);               % returns members of Section
    if ~isempty(member_struct)                         % check if Section is empty
        member_names = fieldnames(member_struct);
        for j=1:length(member_names)
            member_name = char(member_names(j));
            member_value = Structure.(Section).(member_name);
            
            % fix backslashes -- RKM
            for ind = fliplr(find(member_name == '\'))
                member_name = member_name([1:ind ind:end]);
            end
            for ind = fliplr(find(member_value == '\'))
                member_value = member_value([1:ind ind:end]);
            end
            
            fprintf(fid,'%s=%s\n',member_name,member_value); % output member name and value
            
        end % for-END (Members)
    end % if-END
end % for-END (Sections)

fclose(fid); % close file

% Copyright (c) 2009, Dirk Lohse
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.