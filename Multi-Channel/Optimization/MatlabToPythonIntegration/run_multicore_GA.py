
# run_multicore.py
import os, multiprocessing as mp


# run_pipeline.py
import os
import sys
import time
import importlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import Calc_output_grad
import Update_params
import Init_Params
import Calc_output_loss
import Update_params_GA
import Init_Params_GA

from datetime import datetime
from pathlib import Path
from scipy.io import savemat
import numpy as np

# --- Paths (update only if your layout changes) ---
PY_MODULE_DIR = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration"
SOLVE_DIR     = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\1-channel-paper\solve"
PARAMS_FILE   = r"params.mat"  # assumes params.mat is in the CWD; use an absolute path if needed

# Ensure paths are on sys.path (like MATLAB insert/addpath)
for p in [PY_MODULE_DIR, SOLVE_DIR]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# --- Utilities to load MATLAB structs without changing data layout ---
def _todict(matobj):
    """
    Recursively convert MATLAB struct objects (from loadmat) to nested dict/list,
    leaving numeric ndarrays unchanged. Preserves shapes where possible.
    """
    # If it's a size-1 object array, unwrap it
    if isinstance(matobj, np.ndarray) and matobj.dtype == object and matobj.size == 1:
        matobj = matobj[()]

    # MATLAB struct -> Python dict
    if hasattr(matobj, "_fieldnames"):
        d = {}
        for name in matobj._fieldnames:
            d[name] = _todict(getattr(matobj, name))
        return d

    # Object arrays -> list (elementwise conversion)
    if isinstance(matobj, np.ndarray) and matobj.dtype == object:
        return [_todict(x) for x in matobj]

    # Regular ndarrays or scalars -> return as-is
    if isinstance(matobj, np.ndarray):
        return matobj
    else:
        return matobj

def loadmat_struct(filename):
    """
    Load a MAT-file and convert MATLAB structs into nested dict/list without
    altering numeric arrays. This mirrors what your MATLAB->Python bridge delivered.
    """
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = {}
    for k, v in data.items():
        if k.startswith("__"):
            continue
        out[k] = _todict(v)
    return out

def _import_and_reload(module_name):
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    return mod


def save_run(results_dict, out_dir="results"):
    """
    results_dict  : {"losses": ndarray, "output": ndarray, ...}
    out_dir       : folder to hold all .mat files
    """
    Path(out_dir).mkdir(exist_ok=True)

    # e.g. 2025-08-07_09-27-43.mat  (year-month-day_hour-min-sec)
    stamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f_name  = f"run_{stamp}.mat"
    f_path  = Path(out_dir) / f_name

    savemat(f_path, results_dict, do_compression=True)
    print(f"saved results : {f_path.resolve()}")

def run():

    # Inputs that were going to forwards()
    #ps = np.array([1,1,1,1,1,1,1,1,1,1])*0.025          
    scale_factor = 1.0    # whatever you were using

    TRIALS = 10
    N_PROCS = min(TRIALS, os.cpu_count())   # use all cores but don't oversubscribe

     # --- Mirror the import/reload sequence ---
    module_names = [
        "Solve_File_Generator_Manual_GA",
        "Extract_Fixed_vars",
        "Clean_up",
        "State_Parser",
        "State_variable_Identifier",
        "FormatODEs_Ns",
        "ConditionalActions",
        "genPoissonInputs",
        "genPoissonTimes",
        "Calc_output_loss",
        "Update_params_GA",
        "Init_Params_GA",
    ]

    modules = {}
    for name in module_names:
        modules[name] = _import_and_reload(name)


    # --- Load params.mat and extract 'p' without changing its structure ---
    params = loadmat_struct(PARAMS_FILE)
    if "p" not in params:
        raise KeyError(f"'p' not found in {PARAMS_FILE}. Keys present: {list(params.keys())}")
    p = params["p"]

    # --- Build ODEs (same as: ParamsReturned = py.Solve_File_Generator_Manual.build_ODE(p); ) ---
    # If build_ODE returns something you need later, capture it.

    batch_size = 400

    
    ParamsReturned = modules["Solve_File_Generator_Manual_GA"].build_ODE(p,batch_size)

    generated_GA = _import_and_reload("generated_GA")


    from generated_GA_wrapper import _single_trial

    num_epochs = 50
    num_params = 28
    
    #Warning! Make sure params are using Init_Params

    p = Init_Params_GA.pinit(batch_size,num_params,load_from_file=False);
    
    #Initilze GA Hyper-Parameters
    mutation_rate = 0.05 #Change to learn
    mutation_strength = 0.001 #(Learning Rate)
    elite_pop_frac = 0.05 #Keep no matter what

    selection_type = "tournament"
    crossover_type = "random_even"
    mutation_type = "random"


    #1.Forwards
    #2.Selection (Throw out bottom half)
    #3.Reproduction
    #4.Mutation
    #5.Repeat

    #print(np.shape(p))


    scale_factor = 1

    losses = []
    param_tracker = []

    t0 = time.perf_counter()
    
    for epoch in range(num_epochs):

        

        param_tracker.append(p)


        output = []
        grads = np.zeros((10,batch_size))

        with mp.get_context("spawn").Pool(processes=N_PROCS) as pool:
            # Build an argument tuple for each trial
            args_iterable = [(k, p, scale_factor) for k in range(TRIALS)]
        
            # Run trials in parallel and collect their return values
            results = pool.starmap(_single_trial, args_iterable)

            #print(np.shape(results))

           
        for k in range(len(results)):


            output.append(np.array(results[k]))
        
        output = np.stack(output, axis=0)


        #Calculate output loss

        #print(np.shape(output))

        loss = Calc_output_loss.calculate(output,"PSTH")

        p = Update_params_GA.GA_update(p,loss[1],selection_type,crossover_type,mutation_type,mutation_rate,mutation_strength,elite_pop_frac)

        #print(np.shape(p))

        losses.append(loss)
        print(f"Epoch {epoch}: Mean L2 Loss = {np.mean(loss[0])}: Mean L2-PSTH Loss = {np.mean(loss[1])}",flush=True) 
        #print(f"Epoch {epoch}: Mean L2-PSTH Loss = {np.mean(loss[0])}",flush=True) 

    t_post = time.perf_counter() - t0  

    print(f"Sim Time: {t_post:.3f}s")

    return losses, output, param_tracker




if __name__ == "__main__":         

    losses, output, param_tracker = run()   # your training routine

    save_run(
        {
            "losses"        : np.asarray(losses, dtype=np.float32),
            "output"        : np.asarray(output, dtype=np.float32),
            "param_tracker" : np.asarray(param_tracker, dtype=np.float32),
        }
    )

   
