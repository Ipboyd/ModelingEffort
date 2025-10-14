

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
import Init_Params_flexible

from datetime import datetime
from pathlib import Path
from scipy.io import savemat
import numpy as np

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

    return f_path


     



def run():

    # Load in the necessary modules
    module_names = [
        "Solve_File_Generator_Manual_GA_Nlayers",
        "Solve_File_Generator_Manual_Nlayers",
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
        "Init_Params_flexible",
    ]

    modules = {}
    for name in module_names:
        modules[name] = _import_and_reload(name)

    #General Flow:

    #0. Naively estimating the firing rate when there is no stim -- Currently finding this in the GA -- Update this in the future

    #1. Create the solve files for each layer model (1 2 and 3 layers)
    #2. Fit the ISI distribution by adjusting the t_refs for each neuron.
    #3. Fit the PSTH? With Gsyns.
    #4. Repeat for all cells
   
    
    #1. -- Create the solve files
    PY_MODULE_DIR = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Multi-Channel\Optimization\MatlabToPythonIntegration"
    SOLVE_DIR_L23     = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\Layer23\solve"
    SOLVE_DIR_L4     = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\Layer4\solve"
    SOLVE_DIR_L56     = r"C:\Users\ipboy\Documents\GitHub\ModelingEffort\Single-Channel\Model\Model-Core\Model-Main\run\Layer56\solve"
    PARAMS_FILE   = r"params.mat"

    # Ensure paths are on sys.path (like MATLAB insert/addpath)
    for p in [PY_MODULE_DIR, SOLVE_DIR_L23,SOLVE_DIR_L4,SOLVE_DIR_L56]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    

    # --- Load params.mat and extract 'p' without changing its structure ---
    params23 = loadmat_struct(SOLVE_DIR_L23 + '\\' + PARAMS_FILE)
    params4 = loadmat_struct(SOLVE_DIR_L4 + '\\' + PARAMS_FILE)
    params56 = loadmat_struct(SOLVE_DIR_L56 + '\\' + PARAMS_FILE)

    p23 = params23["p"]
    p4 = params4["p"]
    p56 = params56["p"]

    #General Params
    batch_size = 400
    learning_mask = [1,0]
    #GA_Epochs = 10
    Grad_Epochs = 50
    scale_factor = 1
    trials = 10
    n_procs = min(trials, os.cpu_count()) 

    #GA HyperParameteres
    mutation_rate = 0.05 #Change to learn
    mutation_strength = 0.01 #(Learning Rate)
    elite_pop_frac = 0.05 #Keep no matter what

    #Grad Params
    beta1, beta2 = 0.99, 0.9995 
    eps = 1e-6
    lr = 5e-4

    selection_type = "tournament"
    crossover_type = "random_even"
    mutation_type = "random"

    #ParamsReturned = modules["Solve_File_Generator_Manual_GA_Nlayers"].build_ODE(p4,batch_size,'Layer4') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_4_GA = _import_and_reload("generated_GA_Layer4")

    #ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p4,batch_size,learning_mask,'Layer4') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_4 = _import_and_reload("generated_Grad_Layer4")

    #ParamsReturned = modules["Solve_File_Generator_Manual_GA_Nlayers"].build_ODE(p23,batch_size,'Layer23') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_23_GA = _import_and_reload("generated_GA_Layer23")

    #ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p23,batch_size,learning_mask,'Layer23') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_23 = _import_and_reload("generated_Grad_Layer23")

    #ParamsReturned = modules["Solve_File_Generator_Manual_GA_Nlayers"].build_ODE(p56,batch_size,'Layer56') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_56_GA = _import_and_reload("generated_GA_Layer56")

    #ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p56,batch_size,learning_mask,'Layer56') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    #generated_56 = _import_and_reload("generated_Grad_Layer56")

    from generated_wrapper_flexible import _single_trial

    # ---------------- LAYER 4 FIT ----------------------#
    
    # #Grid search for best gain initialization
    # p = Init_Params_flexible.pinit(batch_size,1,"generated_GA_Layer4","",load_from_file=False);

    # #print(p[1])

    # #losses = []
    # #param_tracker = []

    # t0 = time.perf_counter()
    # #for epoch in range(GA_Epochs):
    #     #param_tracker.append(p)
    # output = []

    # with mp.get_context("spawn").Pool(processes=n_procs) as pool:
    #     args_iterable = [(k, p, scale_factor,"generated_GA_Layer4") for k in range(trials)]
    #     results = pool.starmap(_single_trial, args_iterable)

    # for k in range(len(results)):
    #     output.append(np.array(results[k]))
        
    # output = np.stack(output, axis=0)

    # loss = Calc_output_loss.calculate(output,"ISI_Dist")

    # #p = Update_params_GA.GA_update(p,loss,selection_type,crossover_type,mutation_type,mutation_rate,mutation_strength,elite_pop_frac)

    # #losses.append(loss)
    # #print(f"epoch {epoch}: mean l2 loss = {np.mean(loss)}",flush=True) 

    # print(f"ISI-Dist Grid Search: min l2 loss = {np.min(loss)}",flush=True) 

    # t_post = time.perf_counter() - t0  

    # print(f"sim time: {t_post:.3f}s")


    # f_path = save_run(
    #     {
    #         "losses"        : np.asarray(loss, dtype=np.float32),
    #         "output"        : np.asarray(output, dtype=np.float32),
    #         "param_tracker" : np.asarray(p, dtype=np.float32),
    #     }
    # )

    # #print(p4)

    # min_idx = np.argmin(loss)

    # print(np.shape(p))

    # p4['On_On_IC_g_postIC'] = p[0,min_idx]
    # p4['Off_Off_IC_g_postIC'] = p[0,min_idx]


    ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p4,batch_size,learning_mask,'Layer4') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    generated_4 = _import_and_reload("generated_Grad_Layer4")




    num_params = 10
    p = Init_Params_flexible.pinit(batch_size,num_params,"generated_Grad_Layer4");

    #Init mvt
    m = np.zeros((num_params,batch_size))
    v = np.zeros((num_params,batch_size))
    t = 0
    
    losses = []
    param_tracker = []

    t0 = time.perf_counter()
    for epoch in range(Grad_Epochs):
        param_tracker.append(p)
        output = []
        grads = np.zeros((num_params,batch_size))

        with mp.get_context("spawn").Pool(processes=n_procs) as pool:
            # build an argument tuple for each trial
            args_iterable = [(k, p, scale_factor,"generated_Grad_Layer4") for k in range(trials)]
        
            # run trials in parallel and collect their return values
            results = pool.starmap(_single_trial, args_iterable)

           
        for k in range(len(results)):

            grads += np.array(results[k][1])

            output.append(np.array(results[k][0]))
        
        output = np.stack(output, axis=0)

        out_grad, loss = Calc_output_grad.calculate(output, grads, scale_factor, "PSTH")

        m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

        losses.append(loss)
        print(f"epoch {epoch}: mean l2 loss = {np.mean(loss[0])}: mean l2-psth loss = {np.mean(loss[1])}",flush=True) 

    t_post = time.perf_counter() - t0  

    print(f"sim time: {t_post:.3f}s")


    f_path = save_run(
        {
            "losses"        : np.asarray(losses, dtype=np.float32),
            "output"        : np.asarray(output, dtype=np.float32),
            "param_tracker" : np.asarray(param_tracker, dtype=np.float32),
        }
    )


    ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p23,batch_size,learning_mask,'Layer23') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    generated_23 = _import_and_reload("generated_Grad_Layer23")

    num_params = 19
    p = Init_Params_flexible.pinit(batch_size,num_params,"generated_Grad_Layer23");

    #Init mvt
    m = np.zeros((num_params,batch_size))
    v = np.zeros((num_params,batch_size))
    t = 0
    
    losses = []
    param_tracker = []

    t0 = time.perf_counter()
    for epoch in range(Grad_Epochs):
        param_tracker.append(p)
        output = []
        grads = np.zeros((num_params,batch_size))

        with mp.get_context("spawn").Pool(processes=n_procs) as pool:
            # build an argument tuple for each trial
            args_iterable = [(k, p, scale_factor,"generated_Grad_Layer23") for k in range(trials)]
        
            # run trials in parallel and collect their return values
            results = pool.starmap(_single_trial, args_iterable)

           
        for k in range(len(results)):

            grads += np.array(results[k][1])

            output.append(np.array(results[k][0]))
        
        output = np.stack(output, axis=0)

        out_grad, loss = Calc_output_grad.calculate(output, grads, scale_factor, "PSTH")

        m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

        losses.append(loss)
        print(f"epoch {epoch}: mean l2 loss = {np.mean(loss[0])}: mean l2-psth loss = {np.mean(loss[1])}",flush=True) 

    t_post = time.perf_counter() - t0  

    print(f"sim time: {t_post:.3f}s")


    f_path = save_run(
        {
            "losses"        : np.asarray(losses, dtype=np.float32),
            "output"        : np.asarray(output, dtype=np.float32),
            "param_tracker" : np.asarray(param_tracker, dtype=np.float32),
        }
    )

    ParamsReturned = modules["Solve_File_Generator_Manual_Nlayers"].build_ODE(p56,batch_size,learning_mask,'Layer56') #Last Parameter is num_layers -- This could be compessed but for readability why bother
    generated_56 = _import_and_reload("generated_Grad_Layer56")

    num_params = 28
    p = Init_Params_flexible.pinit(batch_size,num_params,"generated_Grad_Layer56");

    #Init mvt
    m = np.zeros((num_params,batch_size))
    v = np.zeros((num_params,batch_size))
    t = 0
    
    losses = []
    param_tracker = []

    t0 = time.perf_counter()
    for epoch in range(Grad_Epochs):
        param_tracker.append(p)
        output = []
        grads = np.zeros((num_params,batch_size))

        with mp.get_context("spawn").Pool(processes=n_procs) as pool:
            # build an argument tuple for each trial
            args_iterable = [(k, p, scale_factor,"generated_Grad_Layer56") for k in range(trials)]
        
            # run trials in parallel and collect their return values
            results = pool.starmap(_single_trial, args_iterable)

           
        for k in range(len(results)):

            grads += np.array(results[k][1])

            output.append(np.array(results[k][0]))
        
        output = np.stack(output, axis=0)

        out_grad, loss = Calc_output_grad.calculate(output, grads, scale_factor, "PSTH")

        m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

        losses.append(loss)
        print(f"epoch {epoch}: mean l2 loss = {np.mean(loss[0])}: mean l2-psth loss = {np.mean(loss[1])}",flush=True) 

    t_post = time.perf_counter() - t0  

    print(f"sim time: {t_post:.3f}s")


    f_path = save_run(
        {
            "losses"        : np.asarray(losses, dtype=np.float32),
            "output"        : np.asarray(output, dtype=np.float32),
            "param_tracker" : np.asarray(param_tracker, dtype=np.float32),
        }
    )


if __name__ == "__main__":         

    run()   # your training routine

    

   
