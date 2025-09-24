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
        "Solve_File_Generator_Manual",
        "Extract_Fixed_vars",
        "Clean_up",
        "State_Parser",
        "State_variable_Identifier",
        "FormatODEs_Ns",
        "ConditionalActions",
        "genPoissonInputs",
        "genPoissonTimes",
        "Calc_output_grad",
        "Update_params",
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

    batch_size = 200

    #Learning mask determines the sets of parameters that you want to learn -- TODO createa a submask so that within each category you can learn just certain connections
    #1 = learn 0 = do not learn
    #[ gsyn ,  tau , more in the future]
    learning_mask = [1,0]
    
    ParamsReturned = modules["Solve_File_Generator_Manual"].build_ODE(p,batch_size,learning_mask)

    generated2 = _import_and_reload("generated2")


    from generated2_wrapper import _single_trial

    num_epochs = 50
    num_params = 10
    
    #Warning! Make sure params are using Init_Params

    p = Init_Params.pinit(batch_size,num_params,learning_mask,load_from_file=False);

    #print(np.shape(p))


    #p = np.transpose(np.array([[0.0014,0.078,0.0609,0.0592,0.0261,0.0125,0.0332,0.0734,0,0.0197],[0.0561,0.0332,0.0613,0.0772,0.0109,0.0499,0.0117,0.0301,0.0449,0.0371]]))

    #p = np.ones((10,batch_size))*np.array([0.025,0.03])#np.array([1,1,1,1,1,1,1,1,1,1])*0.025

    #print(p)
    #p = np.array([0.018998357,	0.028920915,	0.030998280,	0.018925373,	0.031662147,	0.027822275,	0.018823540, 0.031477317,	0.031293292,	0.031775046])
    

    #Initilze Adam Parameters
    m = np.zeros((num_params,batch_size))
    v = np.zeros((num_params,batch_size))
    beta1, beta2 = 0.99, 0.9995   #Nominally 0.92, 0.9995
    eps = 1e-6
    t = 0
    lr = 5e-3

    #print(p)

    #There was an issue where at 1e-7 it looked like the spiking derivative wrt the voltage was misbehaving

    scale_factor = 1

    losses = []
    param_tracker = []

    t0 = time.perf_counter()
    
    for epoch in range(num_epochs):

        

        param_tracker.append(p)


        output = []
        grads = np.zeros((num_params,batch_size))

        with mp.get_context("spawn").Pool(processes=N_PROCS) as pool:
            # Build an argument tuple for each trial
            args_iterable = [(k, p, scale_factor) for k in range(TRIALS)]
        
            # Run trials in parallel and collect their return values
            results = pool.starmap(_single_trial, args_iterable)


        #[]trial[]output/grad[]gradselection/output indicy

        #Trial dimention get appeneded at this level. We should see the results be batch x length
           
        for k in range(len(results)):
            #print(k)
            #print(grads)

            #print(np.array(results[k][1]))

            grads += np.array(results[k][1])

            #print(np.shape(results[k][0]))
            
            #print(results[k][0])

            #if len(output) == 0:
            output.append(np.array(results[k][0]))
            #else:
            #    output = np.stack((output, results[k][0]), axis=0))
        
        output = np.stack(output, axis=0)


        #print(grads)

        

        



        #print(np.shape(np.array(output)))

        # output_single = results[0][0]

        # grads = sum(np.array(results)[:,1])

        # print(grads)

        # if len(output) == 0:
        #     output.append(output_single)
        # else:
        #     output = np.vstack((output, output_single))
            
        #print(output_single)
        #print(grads_single)

        #print(grads_single)
        #print(grads_single2)

        #grads = grads + np.array(grads_single)


        
        #Extract gradients
        #grad_holder2 = []
        #for z in grads:
        #    grad_holder2.append(float(z[0][0]))

        #print(np.shape(np.array(output)))
        #print(grads)


        #calcualtes loss functions
        #---
        #current functions:
        #    - firing rate l2 ("fr")
        #    - psth l2 ("psth")
        #    - spike l2 distance /wip
        #    - van rossum distance (spike level) /wip

        out_grad, loss = Calc_output_grad.calculate(output, grads, scale_factor, "PSTH")

        #Calculate parameter updates using Adam Optimizer
        #---
        #Uses 2 terms to control the momementum of the learning
        #    -beta1 controlls short term momentum
        #    -beta2 contorlls long term dampening




        m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

        #print(np.shape(p))

        #print(f"[epoch {epoch}] starmap: {t_pool:.3f}s   postproc: {t_post:.3f}s   postgrad: {t_post2:.3f}s")

        losses.append(loss)

        #print(losses)
        
        print(f"Epoch {epoch}: Mean L2 Loss = {np.mean(loss[0])}: Mean PSTH Loss = {np.mean(loss[1])}",flush=True) 
        #print(f"Epoch {{epoch}}: Loss = {{loss}}",flush=True) 

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

   
