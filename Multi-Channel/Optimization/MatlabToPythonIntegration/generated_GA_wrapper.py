def _single_trial(trial_number, ps, scale_factor, name):
    """
    Wrapper that runs ONE trial and returns whatever `forwards` returns.
    Put *only* the per-trial work here so we can map it in parallel.
    """
    # forwards() currently runs 10 trials internally; refactor it so
    # that *all* the code inside `for trial_number in range(10): …
    # moves here and forwards() becomes a thin wrapper that calls this.
    #
    #   If you prefer not to touch generated2.py yet, simply copy-paste
    #   the body of that inner loop here.
    #
    import importlib, generated_GA  # your big function
    #import functools
    generated_GA = importlib.reload(generated_GA)

    return generated_GA.main(trial_number, ps, scale_factor)

