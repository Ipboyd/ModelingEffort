function input = df_concat_for_checksum(varargin)
testnum = 0;
done = 0;
while done == 0
    tempfile = fullfile(getenv('TMP'),['Temp_hashing_name_' num2str(testnum) '.tmp']);
    if ~exist(tempfile,'file')
        %save(tempfile,'varargin');
        A = ver('MATLAB');
        A = A.Version;
        A = str2num(A(1));
%         if A > 6
%             to_eval = ['save ' tempfile ' varargin -V6'];
%         else
            to_eval = ['save ' tempfile ' varargin'];
   %     end
        eval(to_eval);
        fid = fopen(tempfile);
        input = fread(fid,inf,'uint8');
        fclose(fid);
        done = 1;
    else
        testnum = testnum + 1;
    end
end
input = uint8(input(120:end));   %  Remove the date stamp in the file; we don't want our hashes to depend on that!
%input = input(find(input));
delete(tempfile);
