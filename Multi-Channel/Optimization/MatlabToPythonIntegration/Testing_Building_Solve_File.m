clear classes

insert(py.sys.path,int32(0),'C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration');

mod = py.importlib.import_module('Solve_File_Generator');
py.importlib.reload(mod);

mod = py.importlib.import_module('Extract_Fixed_vars');
py.importlib.reload(mod);

addpath('C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\1-channel-paper\solve')

params = load('params.mat','p');
p = params.p;

ParamsReturned = py.Solve_File_Generator.build_ODE(p);