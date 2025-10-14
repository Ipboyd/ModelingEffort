import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def plot_g_field(g_field, spike_idx, dt, c=0, b=0, t_ms_window=None, title_suffix=""):
    """
    g_field:   (C,T,B)
    spike_idx: (C,K,B) ints, -1 for no spike
    dt:        ms
    c,b:       which channel & batch to show
    """
    C, T, B = g_field.shape
    t = np.arange(T) * dt
    if t_ms_window is None:
        sl = slice(None)
    else:
        i0 = max(0, int(np.floor(t_ms_window[0] / dt)))
        i1 = min(T, int(np.ceil(t_ms_window[1] / dt)))
        sl = slice(i0, i1)

    # spikes for this (c,b)
    si = spike_idx[c, :, b]
    si = si[(si >= 0) & (si < T)]
    t_spk = si * dt

    # --- g_field trace + spike markers
    plt.figure(figsize=(10, 3.2))
    plt.plot(t[sl], g_field[c, sl, b], label="g_field(t)")
    for tt in t_spk:
        if (sl.start is None or t[sl.start] <= tt) and (sl.stop is None or tt <= t[min(sl.stop-1, T-1)]):
            plt.axvline(tt, linestyle=":", linewidth=0.8)
    plt.title(f"Adjoint field g(t)  [C={c}, B={b}] {title_suffix}")
    plt.xlabel("Time (ms)"); plt.ylabel("g(t)")
    plt.grid(True, alpha=0.3); plt.legend()

    # --- sample g at spikes + histogram
    if si.size:
        g_at_spikes = g_field[c, si, b]
        plt.figure(figsize=(6, 3))
        plt.hist(g_at_spikes, bins=40)
        plt.title(f"Histogram of g(t_k) over spikes  [C={c}, B={b}]")
        plt.xlabel("g(t_k)"); plt.ylabel("# spikes")
        plt.grid(True, alpha=0.3)

        # also overlay as scatter on the trace window
        plt.figure(figsize=(10, 3.2))
        plt.plot(t[sl], g_field[c, sl, b], label="g_field(t)")
        mask_in = (t_spk >= (t[sl.start] if sl.start is not None else t[0])) & \
                  (t_spk <= (t[min(sl.stop-1, T-1)] if sl.stop is not None else t[-1]))
        tsp = t_spk[mask_in]
        gsp = g_at_spikes[mask_in]
        plt.scatter(tsp, gsp, s=20, label="g(t_k) at spikes")
        plt.title(f"g(t) with g(t_k) markers  [C={c}, B={b}]")
        plt.xlabel("Time (ms)"); plt.ylabel("g(t)")
        plt.grid(True, alpha=0.3); plt.legend()

    plt.show()

def plot_vr_traces(e_syn_sim, e_mem_sim, y_sim, y_data, forwards_out, spike_idx, dt,
                   c=0, b=0, t_ms_window=None):
    """
    e_syn_sim, e_mem_sim, y_sim, y_data: arrays shaped (C, T, B)
    forwards_out: (C, T, B) binary spikes (for markers)
    spike_idx: (C, K, B) integer indices; -1 where no spike
    dt: timestep in ms
    c,b: which channel/batch to plot
    t_ms_window: (t0_ms, t1_ms) to zoom; None = full duration
    """
    C, T, B = y_sim.shape
    assert 0 <= c < C and 0 <= b < B
    t = np.arange(T) * dt

    if t_ms_window is not None:
        t0_ms, t1_ms = t_ms_window
        i0 = max(0, int(np.floor(t0_ms / dt)))
        i1 = min(T, int(np.ceil(t1_ms / dt)))
        sl = slice(i0, i1)
    else:
        sl = slice(None)

    # Spike times for vertical markers
    si = spike_idx[c, :, b]
    si = si[(si >= 0) & (si < T)]
    t_spk = si * dt

    # --- One-pole filtered components
    plt.figure(figsize=(10, 3.5))
    plt.plot(t[sl], e_syn_sim[c, sl, b], label='e_syn_sim')
    plt.plot(t[sl], e_mem_sim[c, sl, b], label='e_mem_sim', linestyle='--')
    for tt in t_spk:
        if sl.start is None or (t[sl.start] <= tt <= t[min(sl.stop-1, T-1)]):
            plt.axvline(tt, linestyle=':', linewidth=0.8)
    plt.title(f'One-pole filtered traces (C={c}, B={b})')
    plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3); plt.legend()

    # --- Double-exp output and target + error
    e = (y_sim - y_data)
    plt.figure(figsize=(10, 3.5))
    plt.plot(t[sl], y_sim[c, sl, b], label='y_sim')
    # y_data may have B==1; protect the index
    bd = min(b, y_data.shape[2]-1)
    plt.plot(t[sl], y_data[c, sl, bd], label='y_data', linestyle='--', alpha=0.8)
    plt.plot(t[sl], e[c, sl, b], label='error = y_sim - y_data', linestyle=':')
    for tt in t_spk:
        if sl.start is None or (t[sl.start] <= tt <= t[min(sl.stop-1, T-1)]):
            plt.axvline(tt, linestyle=':', linewidth=0.8)
    plt.title(f'Double-exp output & error (C={c}, B={b})')
    plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.show()

def gather_at_indices(g_field, spike_idx):
    """
    g_field:   (C, T, B)  values defined at every time step
    spike_idx: (C, K, B)  integer indices into the time axis (use -1 for 'no spike')

    returns:
      g_tk: (C, K, B) = g_field sampled at each spike time
      mask: (C, K, B) = True where a valid spike exists
    """
    C, T, B = g_field.shape
    C2, K, B2 = spike_idx.shape
    assert C == C2 and B == B2

    # valid entries: 0..T-1; we’ll zero-out invalid ones later
    mask = (spike_idx >= 0) & (spike_idx < T)
    safe_idx = np.where(mask, spike_idx, 0)  # dummy index for masked spots

    # advanced indexing to gather along time axis
    c = np.arange(C)[:, None, None]        # (C,1,1)
    b = np.arange(B)[None, None, :]        # (1,1,B)
    g_tk = g_field[c, safe_idx, b]         # (C,K,B)

    # zero out padded/invalid spikes
    g_tk = np.where(mask, g_tk, 0.0)
    return g_tk, mask

def reduce_over_C_and_K(g_tk, grads, mask=None, average_over_batch=True):
    """
    g_tk:  (C, K, B)    = L/t_k (the 'adjoint field' sampled at spikes)
    grads: either (C, K, B) or (C, K, B, P) = t_k/
    mask:  (C, K, B) optional; if provided, zeroes non-spikes

    returns:
      out_grad: (B,) if grads is (C,K,B), else (B,P)
                (batch-averaged to (P,) if average_over_batch=True)
    """
    if mask is not None:
        g_tk = np.where(mask, g_tk, 0.0)
        grads = np.where(mask[..., None] if grads.ndim == 4 else mask, grads, 0.0)

    if grads.ndim == 3:                    # no parameter axis
        prod = g_tk * grads                # (C,K,B)
        out = prod.sum(axis=(0, 1))        # (B,)
    else:                                   # grads has parameter axis P
        prod = g_tk[..., None] * grads     # (C,K,B,P)
        out = prod.sum(axis=(0, 1))        # (B,P)

    if average_over_batch:
        return out.mean(axis=0)
    return out


def spike_idx_from_binary(x, thresh=0.5, refrac_ms=0.0, dt=0.1):

    C, T, B = x.shape
    refrac = int(round(refrac_ms / dt))
    per_cb = []
    Kmax = 0

    for c in range(C):
        for b in range(B):
            k = np.flatnonzero(x[c, :, b] > thresh)
            if refrac > 1 and k.size:
                keep = np.empty_like(k, dtype=bool)
                keep[0] = True
                keep[1:] = (np.diff(k) >= refrac)
                k = k[keep]
            per_cb.append(k)
            Kmax = max(Kmax, k.size)

    spike_idx = -np.ones((C, Kmax, B), dtype=np.int32)
    it = 0
    for c in range(C):
        for b in range(B):
            k = per_cb[it]; it += 1
            spike_idx[c, :k.size, b] = k
    return spike_idx


def calculate(forwards_output, grads, scale_factor, grad_type, ncell = None):

    if grad_type == "fr":

        #Set target spiking rate (Hz)
        target_spikes = 100

        #Calulate fr
        output_reshaped = np.reshape(forwards_output,(1,int((35000*scale_factor-2)*10)))
        fr = output_reshaped.sum()/10/(2.9801*scale_factor) #Replace this hardcode with the sim time in seconds TODO

        #Take L2 Loss
        loss = sum((target_spikes-fr)**2)

        #Update the grads
        scale = float(2*(fr - target_spikes))
        out_grad = [scale * g for g in grads] 

        return out_grad, loss

    elif grad_type == "PSTH_old":

        signal_length = 35000
        offset = 3153

        #Bring in target PSTH
        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{matfile_path}/goalPSTH.mat"
        data = scipy.io.loadmat(filename)

        #If only training on part of the signal, scale down the PSTH as well.
        if scale_factor != 1:

            starting_index = round(len(data['ans'][0])*offset/signal_length)
            ending_index = round(len(data['ans'][0])*(offset+scale_factor*signal_length)/signal_length)
            target_spikes = np.array(data['ans'][0][starting_index:ending_index])

        #Sum vertically (Compress to average across trials)
        b = np.sum(forwards_output,axis=0)

        #Calculate the number of bins
        bin_size = int(np.floor(len(b)/len(target_spikes)))
        num_bins = b.shape[0] // bin_size

        #Bin the average
        b_trunc = b[:num_bins * bin_size]
        binned_counts = b_trunc.reshape(num_bins, bin_size).sum(axis=1)

        #Take L2 Loss
        loss = sum((target_spikes-binned_counts)**2)

        #Update the grads
        scale = float(sum(2*(target_spikes-binned_counts)))
        out_grad = [scale * g for g in grads] 

        return out_grad, loss

    elif grad_type == "spikeL2":
        
        # -- Constants

        tau_vr = 10 #ms
        dt = 0.1 #ms
        dts = dt/1000 #in seconds
        alpha = np.exp(-dt/tau_vr)

        # -- Load in data

        filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/picture_fit.mat"
        data = loadmat(filename)['picture']

        forwards_out = np.asarray(forwards_output, dtype=np.float32)  
        data = data.astype(np.float32)

        # -- L2 Loss & Deriv Vectorized

        diff = forwards_out - data[:,:, np.newaxis]  
        L2_loss_avg  = np.mean(np.sum(diff*diff, axis=1),axis=0) 
        L2_deriv_avg = 2.0 * np.mean(np.sum(diff, axis=1),axis=0)

        # -- Vr Loss

        b = [1.0]
        a = [1.0, -alpha]

        traces_sim  = lfilter(b, a, forwards_out, axis=1)
        traces_data = lfilter(b, a, data[:,:, np.newaxis]  , axis=1)


        vr_diff_squared     = (traces_sim - traces_data)**2    
        
        Vr_loss_avg = np.mean(np.trapz(vr_diff_squared, dx=dts, axis=1), axis = 0)

        out_grad = L2_deriv_avg * grads 
        
        return out_grad, [L2_loss_avg, Vr_loss_avg]

    elif grad_type == "vanRossum":


        # -- Constants

        tau_vr1 = 100 #ms
        dt = 0.1 #ms
        dts = dt/1000 #in seconds
        alpha = np.exp(-dt/tau_vr1)

        # -- Load in data

        filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/picture_fit.mat"
        data = loadmat(filename)['picture']

        forwards_out = np.asarray(forwards_output, dtype=np.float32)  
        data = data.astype(np.float32)

        # -- L2 Loss & Deriv Vectorized

        diff = forwards_out - data[:,:, np.newaxis]  
        L2_loss_avg  = np.mean(np.sum(diff*diff, axis=1),axis=0) 

        # -- Vr Loss

        b = [1.0]
        a = [1.0, -alpha]

        traces_sim  = lfilter(b, a, forwards_out, axis=1)
        traces_data = lfilter(b, a, data[:,:, np.newaxis]  , axis=1)

        #Data error
        e = traces_sim-traces_data

        #Derivative of loss w.r.t spike times (most inner part)
        dL_dtk = (1.0/tau_vr1) * traces_sim - forwards_out

        Vr_deriv_avg = np.mean(np.trapz(2.0 * e * dL_dtk, dx=dts, axis=1),axis=0)


        spike_idx = spike_idx_from_binary(forwards_out, thresh=0.5, refrac_ms=1.0, dt=dt)

        plot_g_field(dL_dtk, spike_idx, dt=dt, c=0, b=0, t_ms_window=(500, 900))

        plot_vr_traces(traces_sim, traces_data, e, e,
               forwards_out, spike_idx,
               dt=dt, c=0, b=0, t_ms_window=(500, 800))  # tweak window as you like



        #traces_sim_deriv_heavy  = (1/tau_vr1) * lfilter(b, a, forwards_out, axis=1)
        #traces_data_deriv_heavy = (1/tau_vr1) * lfilter(b, a, data[:,:, np.newaxis]  , axis=1)

        #Basically just multiplying all the positive points by alpha without including any time dependence (this account for edge effects)
        #traces_sim_deriv_delta  = lfilter(b, b, forwards_out, axis=1)
        #traces_data_deriv_delta = lfilter(b, b, data[:,:, np.newaxis]  , axis=1)

        vr_diff_squared     = (traces_sim - traces_data)**2    
        #vr_diff_deriv     = 2 * ((traces_sim_deriv_heavy-traces_sim_deriv_delta) - (traces_data_deriv_heavy-traces_data_deriv_delta))
        
        Vr_loss_avg = np.mean(np.trapz(vr_diff_squared, dx=dts, axis=1), axis = 0)
        #Vr_deriv_avg = np.mean(np.trapz(vr_diff_deriv, dx=dts, axis=1), axis = 0)

        out_grad = Vr_deriv_avg * grads 

        print(Vr_deriv_avg)
        
        return out_grad, [L2_loss_avg, Vr_loss_avg]


    elif grad_type == "SummedVR":

        dt = 0.1  # ms
        dts = dt/1000 #in seconds

        # -- Constants
        tau_mem = 10.0  # ms
        tau_syn = 5.0   # ms
        alpha_mem = np.exp(-dt / tau_mem)
        alpha_syn = np.exp(-dt / tau_syn)
        c = 1.0 / (1.0 - tau_mem / tau_syn)

        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/picture_fit.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:, :, None] 

        # -- L2 Loss
        diff = forwards_out - data
        L2_loss_avg = np.mean(np.sum(diff * diff, axis=1), axis=0)

        # -- forward
        e_syn_sim = lfilter([1.0], [1.0, -alpha_syn], forwards_out, axis=1)
        e_mem_sim = lfilter([1.0], [1.0, -alpha_mem], forwards_out, axis=1)
        y_sim = e_syn_sim - 0.1*e_mem_sim




        e_syn_data = lfilter([1.0], [1.0, -alpha_syn], data, axis=1)
        e_mem_data = lfilter([1.0], [1.0, -alpha_mem], data, axis=1)
        y_data = e_syn_data - 0.1*e_mem_data

        e = y_sim - y_data 


        dL_dtk =  ((1.0/tau_syn) * e_syn_sim - (1.0/tau_mem) * e_mem_sim)

        Vr_deriv_avg = np.mean(np.trapz(2.0 * e * dL_dtk, dx=dts, axis=1),axis=0)

        #print(Vr_deriv_avg)

        #e_rev = e[:, ::-1, :]
        #corr_syn_rev = lfilter([1.0], [1.0, -alpha_syn], e_rev, axis=1)
        #corr_mem_rev = lfilter([1.0], [1.0, -alpha_mem], e_rev , axis=1)
        #corr_syn = corr_syn_rev[:, ::-1, :]
        #corr_mem = corr_mem_rev[:, ::-1, :]


        #g_field = 2.0 * c * ((corr_syn / tau_syn) - (corr_mem / tau_mem))

        


        #spike_idx = spike_idx_from_binary(forwards_out, thresh=0.5, refrac_ms=1.0, dt=dt)

        #plot_g_field(dL_dtk, spike_idx, dt=dt, c=0, b=0, t_ms_window=(500, 900))

        #plot_vr_traces(e_syn_sim, e_mem_sim, y_sim, y_data,
        #       forwards_out, spike_idx,
        #       dt=dt, c=0, b=0, t_ms_window=(500, 800))  # tweak window as you like

        #g_tk, mask = gather_at_indices(g_field, spike_idx)  

        # collapse over channels & spikes to a (B,) scale; normalize by #valid spikes
        #n_eff = np.maximum(mask.sum(axis=(0, 1)), 1)
        #Vr_deriv_batch = (g_tk.sum(axis=(0, 1)) / n_eff) * dt   # (B,)

        # match your existing scalar  grads pattern
        #scale = float(Vr_deriv_batch.mean())


        
        out_grad = Vr_deriv_avg * grads 

        # VR loss (use dx=dt in ms)
        vr_diff_squared = (y_sim - y_data) ** 2
        Vr_loss_avg = np.mean(np.trapz(vr_diff_squared, dx=dts, axis=1), axis=0)

        return out_grad, [L2_loss_avg, Vr_loss_avg]


    elif grad_type == "ISI":


        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = "c:/users/ipboy/documents/github/modelingeffort/multi-channel/plotting/oliverdataplotting/picture_fit.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:,:,None]

        # -- l2 loss
        diff = forwards_out - data
        l2_loss_avg = np.mean(np.sum(diff * diff, axis=1), axis=0)

        # -- ISI

        beta = 10 #Sharpness control
        eps = 1e-8
        dt = 0.1
        dts = dt/1000
        buffer_frac = 0.1
        temp = 10

        data = np.transpose(data,(2,0,1)) #Transpose things to be Batch,trials,timecouse
        forwards_out = np.transpose(forwards_out,(2,0,1))

        # -- Calculate the loss

        breaks = data.astype(bool) 
        breaks_sim = forwards_out.astype(bool)
        T = data.shape[-1]
        T_sim = forwards_out.shape[-1]
        pos = np.arange(T,dtype=int)
        pos_sim = np.arange(T_sim, dtype=int)

        # ---- previous fill (last breakpoint i) ----
        mark = np.where(breaks, pos, 0)              # indices at breakpoints, 0 elsewhere
        mark_sim = np.where(breaks_sim, pos_sim, 0)

        prev_fill = np.maximum.accumulate(mark, axis=-1)
        prev_fill_sim = np.maximum.accumulate(mark_sim, axis = -1)

        # ---- next fill (first breakpoint strictly > i, clamped to last) ----

        arr = np.where(breaks, pos, T)        # sentinel where no break
        arr_sim = np.where(breaks_sim, pos_sim, T_sim)
        right_min = np.minimum.accumulate(arr[..., ::-1], axis=-1)[..., ::-1]  # next  i (or sentinel)
        right_min_sim = np.minimum.accumulate(arr_sim[..., ::-1], axis=-1)[..., ::-1]
        last_break = prev_fill[..., -1]              # rightmost breakpoint per timecourse
        last_break_sim = prev_fill_sim[..., -1] 
        next_ge = np.where(right_min == T, last_break[..., None], right_min)
        next_ge_sim = np.where(right_min_sim == T_sim, last_break_sim[..., None], right_min_sim)
        next_gt = np.concatenate([next_ge[..., 1:], last_break[..., None]], axis=-1)  # shift -> strictly >
        next_gt_sim = np.concatenate([next_ge_sim[..., 1:], last_break_sim[..., None]], axis=-1)  # shift -> strictly >

        # -- Caluclate ISI specifiic metric

        Xisi = next_gt-prev_fill
        Xisi_sim = next_gt_sim-prev_fill_sim


        IofT = np.abs(Xisi_sim-Xisi)/(np.maximum(Xisi_sim,Xisi)+eps)

        #print(np.shape(IofT))

        # -- Take the AUC and calculate mean across trials

        ItAUC = np.trapz(IofT,axis=-1)

        #print(np.shape(ItAUC))

        isi_loss_avg = np.mean(ItAUC,axis=-1) #Loss per object in batch
        
        t = np.arange(-buffer_frac*np.shape(forwards_out)[2]*dts,np.shape(forwards_out)[2]*dts,dts)


        #Build out same thing as below with data

        Bidx, Cidx, Tidx = np.where(mark)

        Tprev = np.zeros(len(Tidx))
        Tprev[1:] = Tidx[:-1]
        is_group_start = np.ones_like(Tidx, dtype=bool)
        is_group_start[1:] = (Bidx[1:] != Bidx[:-1]) | (Cidx[1:] != Cidx[:-1])
        Tprev[is_group_start] = -1
        diff = np.where(Tprev >= 0, Tidx - Tprev, Tidx)
        mark_sim_copy_data = np.zeros((np.shape(mark)[0],np.shape(mark)[1],np.shape(mark)[2]))
        mark_sim_copy_data[Bidx,Cidx,Tidx] = diff

        mark_sim_copy_future_data = np.zeros((np.shape(mark)[0],np.shape(mark)[1],np.shape(mark)[2]))
        Tidx_Future = np.empty_like(Tidx)
        Tidx_Future[1:] = Tidx[:-1]
        Tidx_Future[is_group_start] = 0
        mark_sim_copy_future_data[Bidx,Cidx,Tidx_Future] = diff



        #Trying to build out a difference vector for changes instead of it being biased by the length of the sim.
        Bidx, Cidx, Tidx = np.where(mark_sim)  

        Tprev = np.zeros(len(Tidx))

        Tprev[1:] = Tidx[:-1]

        is_group_start = np.ones_like(Tidx, dtype=bool)
        is_group_start[1:] = (Bidx[1:] != Bidx[:-1]) | (Cidx[1:] != Cidx[:-1])

        Tprev[is_group_start] = -1

        diff = np.where(Tprev >= 0, Tidx - Tprev, Tidx)

        #Create a copy the size of mark_sim for saftey
        mark_sim_copy = np.zeros((np.shape(mark_sim)[0],np.shape(mark_sim)[1],np.shape(mark_sim)[2]))
        mark_sim_copy[Bidx,Cidx,Tidx] = diff
        
        mark_sim_copy_future = np.zeros((np.shape(mark_sim)[0],np.shape(mark_sim)[1],np.shape(mark_sim)[2]))
        Tidx_Future = np.empty_like(Tidx)
        Tidx_Future[1:] = Tidx[:-1]

        Tidx_Future[is_group_start] = 0

        mark_sim_copy_future[Bidx,Cidx,Tidx_Future] = diff


        m1 = (mark_sim_copy != 0).astype(np.int32) 
        m2 = (mark_sim_copy_future != 0).astype(np.int32) 

        kernel_func = 1/(np.exp(-beta*t)+1)
        
        #Local Bump
        #local_bump = beta*(1/(np.exp(-beta*t)+1))*(1-(1/(np.exp(-beta*t)+1)))

        #Delta j+1 is mark_sim_copy_future_data

        #Trying Low di hi minus hi di low over low low (Maybe later?)
        #deriv_kernel_func = ((np.exp(-beta*t)+1)-(beta*np.exp(-beta*t)))/(np.exp(-beta*t)+1)**2

        # #Past Kernels
        # #dervative of future spikes (tk doesn't really exist here). This ends up giving he same time kernel as the non derived version
        # dtf = kernel_func
        # #There are two parts to the tp derivitve. One that has scaling and one that does not dtps does not and is also just the time kernel I think
        # dtp1 = kernel_func
        # #This one has the scaling and everything else
        # dtp2 = -(beta*np.exp(-beta*t))/(np.exp(-beta*t)+1)**2

        #Future Kernels
        #dervative of future spikes (tk doesn't really exist here). This ends up giving he same time kernel as the non derived version
        dtp = kernel_func
        #There are two parts to the tp derivitve. One that has scaling and one that does not dtps does not and is also just the time kernel I think
        dtf1 = -kernel_func
        #This one has the scaling and everything else
        dtf2 = -(beta*np.exp(-beta*t))/(np.exp(-beta*t)+1)**2
        

        #Try to normalize for stability? (dts)

        # kernel = kernel_func*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        # d1_kernel =  dtf*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        # d2_kernel =  dtp1*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        # d3_kernel =  dtp2*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts

        kernel = kernel_func*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        d1_kernel =  dtp*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        d2_kernel =  dtf1*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts
        d3_kernel =  dtf2*np.ones((np.shape(forwards_out)[0],np.shape(forwards_out)[1],len(t))) * dts

        from scipy.signal import fftconvolve

        #print(np.shape(mark_sim))

        t_p = fftconvolve(mark_sim_copy, kernel , mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]
        t_f = fftconvolve(mark_sim_copy_future, kernel , mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]

        t_p_data = fftconvolve(mark_sim_copy_data, kernel , mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]
        t_f_data = fftconvolve(mark_sim_copy_future_data, kernel , mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]

        #print(np.shape(mark_sim_copy))
        #print(np.shape(kernel))
        #print(np.shape(deriv_kernel_func))

        t_p_prime = fftconvolve(m1, d1_kernel , mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]
        t_f1_prime = fftconvolve(m2, d2_kernel, mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]
        t_f2_prime = fftconvolve(mark_sim_copy_future, d3_kernel, mode='full', axes=(-1,))[:,:,int(buffer_frac*np.shape(forwards_out)[2]):int(buffer_frac*np.shape(forwards_out)[2])+np.shape(forwards_out)[2]]
        


        #t_p_prime = t_p1_prime + t_p2_prime
        
        t_f_prime = t_f1_prime + t_f2_prime

        xisim = t_f_data - t_p_data
        xisin = t_f - t_p

        xisin_prime = t_f_prime - t_p_prime

        #Soft L1-ish deriviative
        abs_calc = abs(xisin-xisim)
        #abs_calc = np.sqrt((xisin-xisim)**2) #Pretty sure this is the same thing
        abs_surrogate = ((xisin-xisim)/((np.sqrt((xisin-xisim)**2))+eps)) * xisin_prime

        #Softmax-ish derivative
        soft_calc = np.maximum(xisin,xisim) + eps
        #soft_calc = temp*np.log(np.exp(xisin/temp)+np.exp(xisim/temp))
        soft_surrogate = (1/(np.exp(xisin/temp)+np.exp(xisim/temp))) * np.exp(xisin/temp) * xisin_prime

        I_grad = (soft_calc * abs_surrogate - abs_calc * soft_surrogate)/(soft_calc**2)

        #print('here')
        #print(np.shape(I_grad))

        I_auc = np.trapz(I_grad, axis=-1)

        isi_deriv_avg = np.mean(I_auc, axis=-1)

        #print('grads')
        #print(isi_deriv_avg)


        out_grad = isi_deriv_avg*dts * grads 

        return out_grad, [l2_loss_avg, isi_loss_avg]

        #print(np.shape(y))


        #print(Loss)


        # -- Deriviative


        # - Idea 

        # - Step 1: Create a mark_sim for the future spikes (tf)
        # - Step 2: Create sigmoid representation for all spike times
        # - step 3: Sum across all
        # - step 4: Adjust the gain


        #Start witht the process from mark_sim

        #I think steps 2 and 3 are just a convolution


        # # Create the plot
        # plt.figure(figsize=(10, 5))

        # # Plot tp
        # plt.plot(pos, t_p_prime[0,0,:], label='p_smooth', color='blue')
        # plt.plot(pos, prev_fill_sim[0,0,:], label='tp_sim', color='green')

        # # Plot tf
        # plt.plot(pos, t_f_prime[0,0,:], label='p_smooth', color='blue')
        # plt.plot(pos, next_gt_sim[0,0,:], label='tp_sim', color='green')

        # # Labeling
        # plt.xlabel('Time')
        # plt.ylabel('Signal Amplitude')
        # plt.legend()
        # plt.grid(True)

        # # Show the plot
        # plt.show()


        # # Create the plot
        # plt.figure(figsize=(10, 5))

        # # Plot tp
        # plt.plot(pos, prev_fill[0,0,:], label='tp_data', color='blue')
        # plt.plot(pos, prev_fill_sim[0,0,:], label='tp_sim', color='green')

        # # Plot tf
        # plt.plot(pos, next_gt[0,0,:], label='tf_data', color='orange')
        # plt.plot(pos, next_gt_sim[0,0,:], label='tf_sim', color='red')

        # # Labeling
        # plt.xlabel('Time')
        # plt.ylabel('Signal Amplitude')
        # plt.legend()
        # plt.grid(True)

        # # Show the plot
        # plt.show()

        # #Create the plot
        # plt.figure(figsize=(10, 5))

        # #Plot Xisis
        # plt.plot(pos, Xisi[0,0,:], label='Xisi_data', color='blue')
        # plt.plot(pos, Xisi_sim[0,0,:], label='Xisi_sim', color='green')

        # #Plot IofT
        # plt.plot(pos, IofT[0,0,:], label='IofT', color='orange')

        # #Labeling
        # plt.xlabel('Time')
        # plt.ylabel('Signal Amplitude')
        # plt.legend()
        # plt.grid(True)

        # #Show the plot
        # plt.show()


        #Quick explaination of what is happening

        #sentinel = T

        #Sentinel takes on a value greater than anythin possible within the array
        #Then you filp the array and accumulate the minimas

        #arr[..., ::-1] filps the array
        #then you accumulate along the inner most axis, then you filp back with [..., ::-1]

        #Doing this you end up with the intended trace but just shifted one right basically and there is still some sentinels left over.

        #Last_break looks for the last spike index using prev_fill
        #Then next_ge replaces all of the sentinels with this number
        #Then we shift to the right once.



        


        #inms = []
        #inms_loss = []

        #time_vector = np.arange(0,np.shape(forwards_output)[1])*dts

        
        
        # for sample in range(np.shape(forwards_output)[0]):
        #     tp_sim = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])
        #     tf_sim = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])

        #     tf_data_loss = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])
        #     tf_sim_loss = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])
        #     tp_data_loss = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])
        #     tp_sim_loss = np.zeros([np.shape(forwards_output)[1],np.shape(forwards_output)[2]])

        #     indicies = [np.append(0,np.flatnonzero(forwards_out[sample, :, b] == 1)) for b in range(np.shape(forwards_output)[2])]
        #     indicies2 = np.append(0,np.where(data[sample] == 1)[0])


        #     for b in range(np.shape(forwards_output)[2]):

        #         #-- Calculate ISI Loss

        #         #Sim
        #         future_spikes = indicies[b][1:]*dts
        #         cur_spikes = indicies[b][:-1]*dts

        #         scalar = future_spikes - cur_spikes

        #         mask_p = (0 <= np.subtract.outer((time_vector),(future_spikes)))
        #         tp_sim_loss[:,b] = np.sum(mask_p*scalar,axis=1)

        #         mask_f = mask_p
        #         mask_f[:,1:] = mask_f[:,:-1]
        #         mask_f[:,0] = True
        #         tf_sim_loss[:,b] = np.sum(mask_f*scalar,axis=1)

        #         #Data
        #         future_spikes_data = indicies2[1:]*dts
        #         cur_spikes_data = indicies2[:-1]*dts

        #         scalar_data = future_spikes_data - cur_spikes_data

        #         mask_p = (0 < np.subtract.outer((time_vector),(future_spikes_data)))
        #         tp_data_loss[:,b] = np.sum(mask_p*scalar_data,axis=1)

        #         mask_f = mask_p
        #         mask_f[:,1:] = mask_f[:,:-1]
        #         mask_f[:,0] = True
        #         tf_data_loss[:,b] = np.sum(mask_f*scalar_data,axis=1)

        #         #Calculate Derivative

        #         #deriv = -scalar*np.exp(time_vector)

        #         step_p =  np.sum(((scalar[1:])/((np.exp(-1000*np.subtract.outer((time_vector),(cur_spikes[1:])))+1))),axis=1)
        #         step_f =  np.sum(((scalar[:-1])/((np.exp(-1000*np.subtract.outer((time_vector),(cur_spikes[:-1])))+1))),axis=1)

        #         tf_sim[:,b] = np.sum(((-scalar[:-1]*100*np.exp(-100*np.subtract.outer((time_vector),(cur_spikes[:-1]))))/((np.exp(-100*np.subtract.outer((time_vector),(cur_spikes[:-1])))+1))**2),axis=1)
        #         tp_sim[:,b] = np.sum(((-scalar[1:]*100*np.exp(-100*np.subtract.outer((time_vector),(cur_spikes[1:]))))/((np.exp(-100*np.subtract.outer((time_vector),(cur_spikes[1:])))+1))**2),axis=1)

        #         import matplotlib.pyplot as plt

        #         # Assuming these variables already exist:
        #         # time_vector: 1D array of time points
        #         # tf_sim: 2D array (e.g., time x batch), tf_sim[:, b] is one trace
        #         # tp_sim: 2D array (e.g., time x batch), tp_sim[:, b] is another trace
        #         # b: the batch index you're interested in

        #         # Create the plot
        #         plt.figure(figsize=(10, 5))

        #         # Plot tf_sim
        #         plt.plot(time_vector, step_f, label='tf_sim', color='blue')

        #         # Plot tp_sim
        #         plt.plot(time_vector, step_p, label='tp_sim', color='orange')

        #         # Labeling
        #         plt.xlabel('Time')
        #         plt.ylabel('Signal Amplitude')
        #         plt.title('tf_sim vs tp_sim')
        #         plt.legend()
        #         plt.grid(True)

        #         # Show the plot
        #         plt.show()




        #     alpha_sim = np.array(tf_sim)-np.array(tp_sim)
        #     #alpha_data = np.array(tf_data)-np.array(tp_data)

        #     alpha_sim_loss = np.array(tf_sim_loss)-np.array(tp_sim_loss)
        #     alpha_data_loss = np.array(tf_data_loss)-np.array(tp_data_loss)


        #     denom = np.maximum(np.maximum(alpha_sim_loss, alpha_data_loss), eps)
        #     inm_loss = np.abs(alpha_sim_loss - alpha_data_loss) / denom

        #     #inm_loss = np.abs(alpha_sim_loss-alpha_data_loss)/np.max(alpha_sim_loss,alpha_data_loss,eps)
        #     inms_loss.append(inm_loss)

        #     #print(alpha_sim)
        #     #print(alpha_sim_loss)
        #     #print(inm_loss)

        #     #note -- derived derivate for isi is likely incorrect. you do not need to take the derivative w.r.t the data spikes
        #     #going to try and just leave out the am' terms from the derivation and then trace back and see what happends
        #     #we could do all possible pairwise comparisons (i do not think it is exactly necessary but it could get rid of a little bit of noise in the simulations)
        #     #it would be wise to try both of these options and see what results you get from each of them.
        #     inm = (((alpha_sim_loss-alpha_data_loss)/(np.sqrt((alpha_sim_loss-alpha_data_loss)**2)+eps)) * (alpha_sim))/((1/beta)*((beta*np.exp(beta*alpha_sim))/(np.exp(beta*alpha_sim_loss)+np.exp(beta*alpha_data_loss))))
        #     inms.append(inm)

        # inms_loss = np.stack(inms_loss, axis=0)
        # inms = np.stack(inms, axis=0)

        # avginm_loss = np.mean(inms_loss,axis=0)
        # isi_loss_avg = np.trapz(avginm_loss, axis=0)

        # avginm = np.mean(inms,axis=0)
        # isi_deriv_avg = np.trapz(avginm, axis=0)

        # #print(np.shape(avginm))
        # #print(np.shape(isi_deriv_avg))



    elif grad_type == "PSTH":

        # -- Constants
        dt = 0.1 #ms
        dts = dt/1000 #in seconds
        bin_width = 200 # binwidth/10 = ms ex. binwidth 200 = 20 ms

        # -- Load in data

        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = f"C:/Users/ipboy/Documents/Github/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/PicturesToFit/picture_fit{ncell}contra.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:,:,None]

        data = np.transpose(data,(2,0,1)) #Transpose things to be Batch,trials,timecouse
        forwards_out = np.transpose(forwards_out,(2,0,1))

        # -- L2 Loss & Deriv Vectorized

        diff = forwards_out - data
        L2_loss_avg = np.mean(np.sum(diff * diff, axis=-1), axis=-1)
        #L2_deriv_avg = 2.0 * np.mean(np.sum(diff, axis=1),axis=0)

        # -- PSTH Average

        #print(np.shape(forwards_out))
        #print(np.shape(data))

        #Idea - Transpose creating a seperate dimentions to sum across with the length of the bin width.
        #Going to trim from front of signal because thats the only area of true silence.

        num_bins, remainder = divmod(np.shape(data)[-1], bin_width) 

        forwards_out_r = forwards_out[:,:,remainder:]
        data_r = data[:,:,remainder:]

        #print(np.shape(forwards_out_r))
        #print(np.shape(data_r))

        #print(np.shape(forwards_out_r)[2]/num_bins)


        #print('shapes')
        #print(np.shape(forwards_out_r)[0])
        #print(np.shape(forwards_out_r)[1])
        #print(num_bins)
        #print(bin_width)

        forwards_out_reshaped = forwards_out_r.reshape((np.shape(forwards_out_r)[0],np.shape(forwards_out_r)[1],num_bins,bin_width))
        data_reshaped = data_r.reshape((np.shape(data_r)[0],np.shape(data_r)[1],num_bins,bin_width))

        #print(np.shape(forwards_out_reshaped))
        #print(np.shape(data_reshaped))

        #Histogram per trial
        #Then sum over all trials to get PSTH

        #print(np.shape(forwards_out_reshaped))

        forwards_out_hist = np.sum(np.sum(forwards_out_reshaped,axis=-1),axis=-2)
        data_hist = np.sum(np.sum(data_reshaped,axis=-1),axis=-2)

        import matplotlib.pyplot as plt

        # Assuming these variables already exist:
        # time_vector: 1D array of time points
        # tf_sim: 2D array (e.g., time x batch), tf_sim[:, b] is one trace
        # tp_sim: 2D array (e.g., time x batch), tp_sim[:, b] is another trace
        # b: the batch index you're interested in


        #plt.figure(figsize=(10, 5))

        #print(np.shape(data_hist))
        #print(np.shape(forwards_out_hist))

        #print(data_hist)

        # Plot tf_sim

        #plt.plot(data_hist[0,:], label='sim_PSTH')


        # Plot tp_sim

        #plt.show()
              
        #print(np.shape(forwards_out_hist))
        #print(np.shape(data_hist))

        diff = forwards_out_hist - data_hist

        #print(np.shape(diff))
 
        #PSTH_loss_avg = np.mean(np.sum(diff * diff, axis=-1), axis=-1)
        PSTH_loss_avg = np.sum(diff * diff, axis=-1)
        #Try MSE instead of SSE
        #PSTH_loss_avg = np.mean(diff * diff, axis=-1)

        #print(PSTH_loss_avg)

        #print(np.shape(PSTH_loss_avg))

        #PSTH_deriv_avg = 2.0 * np.mean(np.sum(diff, axis=-1),axis=-1)
        PSTH_deriv_avg = 2.0 * np.sum(diff, axis=-1)
        
        out_grad = PSTH_deriv_avg * grads 

        return out_grad, [L2_loss_avg, PSTH_loss_avg]


    elif grad_type == "PSTH_VR":

        # -- Constants
        tau_vr1 = 100 #ms
        dt = 0.1 #ms
        dts = dt/1000 #in seconds
        bin_width = 200 # binwidth/10 = ms ex. binwidth 200 = 20 ms
        alpha = np.exp(-dt/tau_vr1)
        loss_gain = 50 # attempt to keep the loss values fairly close in magnetude

        # -- Load in data

        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = "c:/users/ipboy/documents/github/modelingeffort/multi-channel/plotting/oliverdataplotting/picture_fit.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:,:,None]


        # -- VR additional deriv

        b = [1.0]
        a = [1.0, -alpha]

        traces_sim  = lfilter(b, a, forwards_out, axis=1)
        traces_data = lfilter(b, a, data  , axis=1)

        #Data error
        e = traces_sim-traces_data

        #Derivative of loss w.r.t spike times (most inner part)
        dL_dtk = (1.0/tau_vr1) * traces_sim - forwards_out

        Vr_deriv_avg = np.mean(np.trapz(2.0 * e * dL_dtk, dx=dts, axis=1),axis=0)


        data = np.transpose(data,(2,0,1)) #Transpose things to be Batch,trials,timecouse
        forwards_out = np.transpose(forwards_out,(2,0,1))

        # -- L2 Loss & Deriv Vectorized

        diff = forwards_out - data
        L2_loss_avg = np.mean(np.sum(diff * diff, axis=-1), axis=-1)

        # -- PSTH Average

        num_bins, remainder = divmod(np.shape(data)[-1], bin_width) 

        forwards_out_r = forwards_out[:,:,remainder:]
        data_r = data[:,:,remainder:]

        forwards_out_reshaped = forwards_out_r.reshape((np.shape(forwards_out_r)[0],np.shape(forwards_out_r)[1],num_bins,bin_width))
        data_reshaped = data_r.reshape((np.shape(data_r)[0],np.shape(data_r)[1],num_bins,bin_width))

        forwards_out_hist = np.sum(np.sum(forwards_out_reshaped,axis=-1),axis=-2)
        data_hist = np.sum(np.sum(data_reshaped,axis=-1),axis=-2)

        diff = forwards_out_hist - data_hist


        PSTH_loss_avg = np.sum(diff * diff, axis=-1)
        PSTH_deriv_avg = 2.0 * np.sum(diff, axis=-1)

        

        #print(PSTH_deriv_avg)
        #print(Vr_deriv_avg)
        
        out_grad = (PSTH_deriv_avg + loss_gain*Vr_deriv_avg) * grads 

        return out_grad, [L2_loss_avg, PSTH_loss_avg]

    else:
        print("please enter valid loss type")
        return
                    #tf_sim_loss_sum = np.sum)



                #for k in range(np.shape(forwards_output)[1]):
                #    tf_data_sum = 0
                #    tf_sim_sum = 0
                #    tp_data_sum = 0
                #    tp_sim_sum = 0

                #    tf_data_loss_sum = 0
                #    tf_sim_loss_sum = 0
                #    tp_data_loss_sum = 0
                #    tp_sim_loss_sum = 0




                #     for m in range(len(indicies[b])-1):
                #         #divide by 10000 to get to our dt timestep

                #         #print(len(indicies))
                #         #print(len(indicies[b]))

                #         tf_scale = -((indicies[b][m+1]/10000)-(indicies[b][m]/10000))

                #         if m > 0:
                #             tp_scale = -((indicies[b][m]/10000)-(indicies[b][m-1]/10000))
                #         else:
                #             tp_scale = 0
                        

                #         tf_sim_sum += (tf_scale*np.exp(-((k/10000)-(indicies[b][m+1]/10000))))/((np.exp(-((k/10000)-(indicies[b][m+1]/10000)))+1))**2
                #         tp_sim_sum += (tp_scale*np.exp(-((k/10000)-(indicies[b][m]/10000))))/((np.exp(-((k/10000)-(indicies[b][m]/10000)))+1))**2

                #         if indicies[b][m] < k:

                #             tf_sim_loss_sum += indicies[b][m+1]
                #             tp_sim_loss_sum += indicies[b][m]

                #     for m2 in range(len(indicies2)-1):

                #         # tf_scale = -((indicies2[m2+1]/10000)-(indicies2[m2]/10000))

                #         # if m > 0:
                #         #     tp_scale = -((indicies2[m2]/10000)-(indicies2[m2-1]/10000))
                #         # else:
                #         #     tp_scale = 0

                #         # tf_data_sum += (tf_scale/np.exp(-((k/10000)-(indicies2[m2+1]/10000))))/(1/(np.exp(-((k/10000)-(indicies2[m2+1]/10000)))+1))**2
                #         # tp_data_sum += (tp_scale/np.exp(-((k/10000)-(indicies2[m2]/10000))))/(1/(np.exp(-((k/10000)-(indicies2[m2]/10000)))+1))**2

                #         if indicies2[m2] < k:

                #             tf_data_loss_sum += indicies2[m2+1]
                #             tp_data_loss_sum += indicies2[m2]

                #     tf_data_batch = np.append(tf_data_batch,tf_data_sum)
                #     tf_sim_batch = np.append(tf_sim_batch,tf_sim_sum)
                #     tp_data_batch = np.append(tp_data_batch,tp_data_sum)
                #     tp_sim_batch = np.append(tp_sim_batch,tp_sim_sum)

                #     tf_data_loss_batch = np.append(tf_data_loss_batch,tf_data_loss_sum)
                #     tf_sim_loss_batch = np.append(tf_sim_loss_batch,tf_sim_loss_sum)
                #     tp_data_loss_batch = np.append(tp_data_loss_batch,tp_data_loss_sum)
                #     tp_sim_loss_batch = np.append(tp_sim_loss_batch,tp_sim_loss_sum)

                # tf_data.append(tf_data_batch)
                # tf_sim.append(tf_sim_batch)
                # tp_data.append(tp_data_batch)
                # tp_sim.append(tp_sim_batch)

                # tf_data_loss.append(tf_data_loss_batch)
                # tf_sim_loss.append(tf_sim_loss_batch)
                # tp_data_loss.append(tp_data_loss_batch)
                # tp_sim_loss.append(tp_sim_loss_batch)