import numpy as np
import scipy.io

def calculate(fowards_output, grads, scale_factor, grad_type):

    if grad_type == "fr":

        #Set target spiking rate (Hz)
        target_spikes = 100

        #Calulate fr
        output_reshaped = np.reshape(fowards_output,(1,int((35000*scale_factor-2)*10)))
        fr = output_reshaped.sum()/10/(2.9801*scale_factor) #Replace this hardcode with the sim time in seconds TODO

        #Take L2 Loss
        loss = sum((target_spikes-fr)**2)

        #Update the grads
        scale = float(2*(fr - target_spikes))
        out_grad = [scale * g for g in grads] 

        return out_grad, loss

    elif grad_type == "PSTH":

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
        b = np.sum(fowards_output,axis=0)

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
        return 0



    elif grad_type == "vanRossum":


        #Set parameter 
        tau = 10 #(ms)


        #TODO Bring in the target spikes
        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{matfile_path}/150R_230201_ks2_5_ctrl_column_1_chan20.mat"
        data = scipy.io.loadmat(filename)

        print(data)
        print('here')

        print(np.shape(data))
        print(np.shape(fowards_output),flush=True)

        #Reshape the data
        #output_reshaped = np.reshape(fowards_output,(1,int((29801*scale_factor-2)*10)))
        data_reshpaed = '\TODO'

        return out_grad, loss




    else:
        print("please enter valid loss type")
        return
