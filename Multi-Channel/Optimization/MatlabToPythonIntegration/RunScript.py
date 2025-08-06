# run_pipeline.py
import os
import sys
import time
import importlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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

def main():
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
    ParamsReturned = modules["Solve_File_Generator_Manual"].build_ODE(p)

    # --- Import generated2 and run main (like py.generated2.main()) ---
    generated2 = _import_and_reload("generated2")

    t0 = time.perf_counter()
    x = generated2.main()
    t1 = time.perf_counter()

    print(f"Elapsed: {t1 - t0:.3f} s")
    return x, ParamsReturned

if __name__ == "__main__":
    # Optional: set working directory explicitly if you relied on MATLAB's CWD
    # os.chdir(SOLVE_DIR)  # uncomment if generated2 or others assume this as CWD
    out, params_returned = main()