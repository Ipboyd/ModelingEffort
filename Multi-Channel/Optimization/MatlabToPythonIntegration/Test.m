% py.importlib.import_module('my_script');
% result = py.my_script.greet('Isaac');
% disp(string(result))

% pairs = py.Parser.extract_rhs_lhs('solve_ode_1_channel_paper.m');
% 
% n = length(pairs);
% S = struct('lhs', cell(1, n), 'rhs', cell(1, n));
% 
% for i = 1:n
%     S(i).lhs = string(pairs{i}(1));  % Left-hand side variable
%     S(i).rhs = string(pairs{i}(2));  % Right-hand side expression
% end
% 
% jsonText = jsonencode(S);
% 
% fid = fopen('rhs_lhs_pairs.json', 'w');
% fwrite(fid, jsonText, 'char');
% fclose(fid);

%py.importlib.invalidate_caches();
%py.sys.modules('say_hello').remove;  

clear classes

insert(py.sys.path,int32(0),'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration');

mod = py.importlib.import_module('Solve_File_Generator');
py.importlib.reload(mod);

addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\1-channel-paper\solve')

params = load('params.mat','p');
p = params.p;
% 
ParamsReturned = py.Solve_File_Generator.build_ODE(p);
% 
% char(ParamsReturned)

% Add path to your Python file (optional if already on Python path)
% Example: if hello.py is in 'C:\Users\yourname\Documents\MATLAB'
% then set the Python path accordingly

% Call the function
%msg = py.hello.say_hello();

% Convert Python string to MATLAB string and display it
%disp(char(msg));