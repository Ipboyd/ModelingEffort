import numpy as np
import torch

def gen_poisson_times(N_pop, dt, FR, std, simlen=35000):
    """
    Generate Poisson spike trains with refractory period.

    Parameters:
        N_pop : int
            Number of neurons in population.
        dt : float
            Time step in ms.
        FR : float
            Firing rate in Hz.
        std : float
            Standard deviation of the firing rate.
        simlen : int
            Number of time steps (default = 35000)

    Returns:
        token : torch.Tensor
            Binary spike train matrix of shape (simlen, N_pop)
    """
    # Generate Poisson spikes with added noise
    rand_gauss = FR + std * np.random.randn(simlen, N_pop)
    rand_bin = np.random.rand(simlen, N_pop) < (rand_gauss * dt / 1000)

    temp = rand_bin.astype(np.uint8)
    refrac = 1.0  # ms

    for i in range(N_pop):
        spk_inds = np.where(temp[:, i])[0]
        if len(spk_inds) > 1:
            ISIs = np.diff(spk_inds) * dt
            violate_inds = np.where(ISIs < refrac)[0] + 1
            temp[spk_inds[violate_inds], i] = 0

    return torch.tensor(temp, dtype=torch.float32)
