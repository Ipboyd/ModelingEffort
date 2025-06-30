clear classes



insert(py.sys.path,int32(0),'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration');

mod = py.importlib.import_module('Solve_File_Generator');
py.importlib.reload(mod);

mod = py.importlib.import_module('Extract_Fixed_vars');
py.importlib.reload(mod);

mod = py.importlib.import_module('Clean_up');
py.importlib.reload(mod);

mod = py.importlib.import_module('State_Parser');
py.importlib.reload(mod); 

mod = py.importlib.import_module('State_variable_Identifier');
py.importlib.reload(mod);

mod = py.importlib.import_module('FormatODEs_Ns');
py.importlib.reload(mod);

mod = py.importlib.import_module('ConditionalActions');
py.importlib.reload(mod);

mod = py.importlib.import_module('genPoissonInputs');
py.importlib.reload(mod);

mod = py.importlib.import_module('genPoissonTimes');
py.importlib.reload(mod);


addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\1-channel-paper\solve')

params = load('params.mat','p');
p = params.p;



ParamsReturned = py.Solve_File_Generator.build_ODE(p);

mod = py.importlib.import_module('generated');z
py.importlib.reload(mod);

%py.sys.path    
tic
% xs = [];
% for k = 1:1
%    xs = [xs;single(py.generated.main(py.int(k)))];
% end
x = py.generated.main(); %This is declared in python rn so the 1 isn't actually doing anything


toc