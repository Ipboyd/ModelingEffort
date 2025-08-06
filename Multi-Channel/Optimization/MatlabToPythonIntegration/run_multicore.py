# run_multicore.py
import os, multiprocessing as mp
from generated2_wrapper import _single_trial

# Inputs that were going to forwards()
ps = [...]            # list or np.ndarray, unchanged
scale_factor = 1.0    # whatever you were using

TRIALS = 10
N_PROCS = min(TRIALS, os.cpu_count())   # use all cores but don't oversubscribe

if __name__ == "__main__":              # required on Windows!
    with mp.get_context("spawn").Pool(processes=N_PROCS) as pool:
        # Build an argument tuple for each trial
        args_iterable = [(k, ps, scale_factor) for k in range(TRIALS)]
        
        # Run trials in parallel and collect their return values
        results = pool.starmap(_single_trial, args_iterable)

    # ------------------------------------------------------------
    # 3.  Post-process / compile the results any way you like
    # ------------------------------------------------------------
    # Example – stack spike trains or metrics that each trial returns
    # combined = np.stack(results)   # if every trial returns the same-shaped array
    #
    # Example – write each trial’s output to disk from the worker    (faster,
    #                                                                less RAM)
    #
    # Example – pickle/dill each result object here:
    # import pickle, pathlib
    # outdir = pathlib.Path("trial_outputs"); outdir.mkdir(exist_ok=True)
    # for k, res in enumerate(results):
    #     pickle.dump(res, open(outdir/f"trial_{k}.pkl", "wb"))
