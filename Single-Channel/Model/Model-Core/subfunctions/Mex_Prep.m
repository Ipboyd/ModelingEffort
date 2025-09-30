

if isfolder(study_dir)
    warning('off', 'MATLAB:rmpath:DirNotFound');  % suppress path-related warnings
    rmpath(genpath(study_dir));                   % remove study_dir + subfolders from path
    rmdir(study_dir, 's');                         % safely remove folder
    warning('on', 'MATLAB:rmpath:DirNotFound');
end

solve_directory = fullfile(study_dir, 'solve');

if Mex_option == 1
    if exist(fullfile(study_dir, 'solve'), 'dir')
        %don't remove the directory
        flag_raised_mex = 1;
    else
        if exist(study_dir, 'dir'), msg = rmdir(study_dir, 's'); end
        mkdir(solve_directory); 
        flag_raised_mex  = 0;
    
        mexes_dir = fullfile(mfiledir{1:end-1}, 'mexes');
        if isfolder(mexes_dir)
            %These might need to be changed for this model!!!!
            m_file_to_copy = 'solve_ode_1_channel_paper.m';
            mex_file_to_copy = 'solve_ode_1_channel_paper_mex.mexw64';
            mex_file_path = fullfile(mexes_dir, mex_file_to_copy);
            mex_files = dir([mex_file_path, '.*']);
            if ~isempty(mex_files)
                flag_raised_mex = 1;
    
                end
            end
    end
end

addpath(solve_directory);