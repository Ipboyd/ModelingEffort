import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

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
