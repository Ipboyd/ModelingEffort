clear classes

% add folder to Python path once
insert(py.sys.path,int32(0), ...
       'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration');

mod = py.importlib.import_module('run_multicore');   % load / reload the module
py.importlib.reload(mod);                            % pick up recent edits

% simplest call: use all defaults
out = mod.run();                                     % returns Python list


