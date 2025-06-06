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

    #Convert things from tensors so we can work with them
    FR = FR.numpy().astype(int)
    std = std.numpy().astype(int)
    simlen = simlen.numpy().astype(int)
    dt = dt.numpy().astype(int)

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


'''import torch

def gen_poisson_times(N_pop, dt, FR, std, simlen=35000, device=None):
    """
    Generate Poisson spike trains with a refractory period using PyTorch (GPU compatible).

    Parameters:
        N_pop : int
            Number of neurons in population.
        dt : float or torch scalar
            Time step in ms.
        FR : float or torch scalar
            Firing rate in Hz.
        std : float or torch scalar
            Standard deviation of the firing rate.
        simlen : int or torch scalar
            Number of time steps (default = 35000)
        device : torch.device or None
            Device to place tensors on (default: autodetect CUDA)

    Returns:
        token : torch.Tensor
            Binary spike train matrix of shape (simlen, N_pop) on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to Python scalars if they are tensors
    if isinstance(FR, torch.Tensor): FR = FR.item()
    if isinstance(std, torch.Tensor): std = std.item()
    if isinstance(simlen, torch.Tensor): simlen = int(simlen.item())
    if isinstance(N_pop, torch.Tensor): N_pop = int(N_pop.item())
    if isinstance(dt, torch.Tensor): dt = dt.item()

    # Generate Gaussian-modulated firing rate
    rand_gauss = FR + std * torch.randn(simlen, N_pop, device=device)
    rand_bin = torch.rand(simlen, N_pop, device=device) < (rand_gauss * dt / 1000)

    # Convert to uint8-like tensor (float32 binary)
    temp = rand_bin.float()

    # Apply refractory period of 1 ms
    refrac = 1.0  # ms
    refrac_steps = int(refrac / dt)

    if refrac_steps > 0:
        for i in range(N_pop):
            spike_times = torch.where(temp[:, i] > 0)[0]
            if spike_times.numel() > 1:
                ISIs = spike_times[1:] - spike_times[:-1]
                violate_mask = ISIs < refrac_steps
                violate_inds = spike_times[1:][violate_mask]
                temp[violate_inds, i] = 0.0

    return temp  # shape: (simlen, N_pop) on correct device'''
