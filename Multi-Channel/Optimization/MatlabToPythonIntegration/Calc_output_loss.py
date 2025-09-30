import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def movmean_centered(x, w, axis=-1):
            x = np.asarray(x, dtype=float)
            if w < 1 or int(w) != w:
                raise ValueError("w must be a positive integer")
            k = np.ones(int(w), dtype=float)

            def _conv1d(arr):
                num = np.convolve(arr, k, mode='same')
                den = np.convolve(np.ones_like(arr), k, mode='same')  # window size per position
                return num / den

            return np.apply_along_axis(_conv1d, axis, x)

def per_batch_hist_fixed_width(collapsed, start, width, nbins):
    B = collapsed.shape[0]
    out = np.zeros((B, nbins), dtype=np.int64)
    for b, x in enumerate(collapsed[:, 0]):
        if not x.size: 
            continue
        idx = np.floor((x - start) / width).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < nbins)]
        out[b] = np.bincount(idx, minlength=nbins)
    return out

def calculate(forwards_output, grad_type):

    if grad_type == "PSTH":

        # -- Constants
        dt = 0.1 #ms
        dts = dt/1000 #in seconds
        bin_width = 200 # binwidth/10 = ms ex. binwidth 200 = 20 ms

        # -- Load in data

        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = "c:/users/ipboy/documents/github/modelingeffort/multi-channel/plotting/oliverdataplotting/picture_fit.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:,:,None]

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

        return [L2_loss_avg, PSTH_loss_avg]


    if grad_type == "ISI_Dist":

        # -- Calculate the ISI dist for the data
        forwards_out = np.asarray(forwards_output, dtype=np.float32)
        filename = "c:/users/ipboy/documents/github/modelingeffort/multi-channel/plotting/oliverdataplotting/picture_fit.mat"
        data = loadmat(filename)['picture'].astype(np.float32)[:,:,None]


        #-- Calc histogram for the data

        T, N, B = data.shape  # trial, time, batch
        diffs = np.empty((T, B), dtype=object)

        #print(a)

        for t in range(T):
            for bch in range(B):
                times = np.flatnonzero(data[t, :, bch])  # time indices with 1s
                diffs[t, bch] = np.diff(times)        # empty if <2 ones


        #print(diffs)
        # diffs: shape (T, B), dtype=object, each entry is a 1-D np.array of ints
        T, B = diffs.shape

        # Option A: straightforward loop -> (B, 1) object array
        collapsed = np.empty((B, 1), dtype=object)
        for b in range(B):
            # np.concatenate handles empty arrays fine
            collapsed[b, 0] = np.concatenate(diffs[:, b].tolist())

        # Option B: compact one-liner (same result)
        collapsed = np.array([[np.concatenate(diffs[:, b].tolist())] for b in range(B)], dtype=object)

        # collapsed: shape (B, 1), each entry is a 1-D int array
        B = collapsed.shape[0]
        gmax = max((x.max() if x.size else -1) for x in collapsed[:, 0])
        nbins = gmax + 1

        #print(collapsed)

        counts_by_batch = np.zeros((B, nbins), dtype=np.int64)
        #for b, x in enumerate(collapsed[:, 0]):
        #    if x.size:
        #        counts_by_batch[b] = np.bincount(x.astype(np.int64, copy=False), minlength=nbins)

        counts_by_batch_data = per_batch_hist_fixed_width(collapsed,start = 0.0, width = 10.0, nbins = 50)

        #print(np.shape(counts_by_batch_data))

        #-- Calc histogram for the simulation

        T, N, B = forwards_out.shape  # trial, time, batch
        diffs = np.empty((T, B), dtype=object)

        #print(a)

        for t in range(T):
            for bch in range(B):
                times = np.flatnonzero(forwards_out[t, :, bch])  # time indices with 1s
                diffs[t, bch] = np.diff(times)        # empty if <2 ones


        #print(diffs)
        # diffs: shape (T, B), dtype=object, each entry is a 1-D np.array of ints
        T, B = diffs.shape

        # Option A: straightforward loop -> (B, 1) object array
        collapsed = np.empty((B, 1), dtype=object)
        for b in range(B):
            # np.concatenate handles empty arrays fine
            collapsed[b, 0] = np.concatenate(diffs[:, b].tolist())

        # Option B: compact one-liner (same result)
        collapsed = np.array([[np.concatenate(diffs[:, b].tolist())] for b in range(B)], dtype=object)

        # collapsed: shape (B, 1), each entry is a 1-D int array
        B = collapsed.shape[0]
        gmax = max((x.max() if x.size else -1) for x in collapsed[:, 0])
        nbins = gmax + 1

        #print(collapsed)

        counts_by_batch = np.zeros((B, nbins), dtype=np.int64)
        #for b, x in enumerate(collapsed[:, 0]):
        #    if x.size:
        #        counts_by_batch[b] = np.bincount(x.astype(np.int64, copy=False), minlength=nbins)

        counts_by_batch_sim = per_batch_hist_fixed_width(collapsed,start = 0.0, width = 10.0, nbins = 50)   #Try less bins so that we are focused closer to the acutal refractory periods

        #print(np.shape(counts_by_batch_sim))

        diff = counts_by_batch_sim - counts_by_batch_data

        #print(np.shape(diff))

        L2_loss_avg = np.mean((diff * diff), axis=-1)


        #print(np.shape(L2_loss_avg))


        return L2_loss_avg











        #print(counts_by_batch)

        # usage (counts_by_batch is (B, nbins); smooth along bins):
        #smooth = movmean_centered(counts_by_batch, w=1, axis=1)

        #print(smooth)

        #import matplotlib.pyplot as plt

        #plt.plot(smooth[0,:])
        #plt.show()


        #print()

        # -- Calculate the ISI dist for the simulations