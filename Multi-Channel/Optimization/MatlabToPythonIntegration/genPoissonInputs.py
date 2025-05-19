import numpy as np
import scipy.io

def spike_generator(rate, dt, t_ref, t_ref_rel, rec):
    """
    Generate a Poisson spike train with an absolute and relative refractory period.
    """
    dt_sec = dt / 1000  # ms to seconds
    n = len(rate)
    spike_train = np.zeros(n)
    spike_times = []

    n_refab = int(15 / 1000 / dt_sec)  # number of samples for ref. period window
    tw = np.arange(n_refab + 1)

    t_ref_samp = t_ref / 1000 / dt_sec
    t_rel_samp = t_ref_rel / 1000 / dt_sec

    # Recovery function based on Schaette et al. 2005
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.power(tw - t_ref_samp, rec) / (
            np.power(tw - t_ref_samp, rec) + np.power(t_rel_samp, rec)
        )
        w[tw < t_ref_samp] = 0
        w = np.nan_to_num(w)

    x = np.random.rand(n)

    for i in range(n):
        if spike_times and i - spike_times[-1] < n_refab:
            rate[i] *= w[i - spike_times[-1]]
        if x[i] < dt_sec * rate[i]:
            spike_train[i] = 1
            spike_times.append(i)

    return spike_train


def gen_poisson_inputs(trial, loc_num, label, t_ref, t_ref_rel, rec, matfile_path):
    """
    Generate Poisson spike inputs from a .mat file of spike rates.

    Parameters:
        trial : int
            Trial index (1-based in MATLAB, 0-based in Python).
        loc_num : int or None
            Location number, or None to use all.
        label : str
            Label for the stimulus (e.g. 'on' or 'off').
        t_ref, t_ref_rel : float
            Absolute and relative refractory periods (ms).
        rec : float
            Sharpness of relative refractory function.
        matfile_path : str
            Path to the IC_spks_TYPE.mat file.

    Returns:
        s : np.ndarray
            Binary spike train matrix (time x neurons)
    """
    dt = 0.1  # ms
    label_clean = label.strip("'")  # remove literal apostrophes
    filename = f"{matfile_path}/IC_spks_{label_clean}.mat"
    
    data = scipy.io.loadmat(filename)
    temp = data['spks']  # shape: (time * locations, neurons, trials)

    loc_size = temp.shape[0] // 24
    trial_rate = temp[:, :, trial]  # select trial

    if loc_num is not None:
        rate = trial_rate[loc_size * (loc_num - 1):loc_size * loc_num, :]
    else:
        rate = trial_rate

    s = np.zeros_like(rate)

    for n in range(rate.shape[1]):
        s[:, n] = spike_generator(rate[:, n], dt, t_ref, t_ref_rel, rec)

    return s

