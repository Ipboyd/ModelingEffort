import numpy as np
import scipy.io

def calculate(forwards_output, grads, scale_factor, grad_type):

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
        
        #Set parameter 
        tau = 10 #(ms)

        #TODO Bring in the target spikes
        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{matfile_path}/picture_fit.mat"
        data = scipy.io.loadmat(filename)


        data = data['picture']
        L2_deriv_avg = [];
        L2_loss_avg = [];
        Vr_Loss_avg = [];

       

        #Loop through every trial
        for m in range(10):


            vr_loss = 0;
            L2_deriv = 0;
            L2_loss = 0;
            vr_data = []
            


            #Go through the picture horizontally
            data_trial = data[m]
            sim_trial = np.array(forwards_output[m])

            data_spikes = np.where(data_trial == 1)[0]
            sim_spikes = np.where(sim_trial == 1)[0]


            for k in range(np.shape(data)[1]):

                #print('data length')
                #print(np.shape(data)[1])

                L2_deriv += 2*(sim_trial[k]-data_trial[k])
                L2_loss += (sim_trial[k]-data_trial[k])**2

                data_val = 0
                data_deriv = 0
                for z in range(len(data_spikes)):

                    if data_spikes[z] <= k:
                        #Convert indicies and time constant to be in seconds
                        data_val += np.exp(-((k-data_spikes[z])/10000)/(tau/1000))

                    else:
                        data_val += 0


                #Find g(t)
                sim_val = 0


                for z in range(len(sim_spikes)):
                    if sim_spikes[z] <= k:
                        #Convert indicies and time constant to be in seconds
                        sim_val += np.exp(-((k-sim_spikes[z])/10000)/(tau/1000))


                    else:
                        sim_val += 0

                vr_data.append((sim_val-data_val)**2)
            

            #Take the auc of vr_data and vr_deriv
            for j in range(len(vr_data)-1):
                #0.1/1000 is dt in seconds (slice width)
                vr_loss += (vr_data[j]+vr_data[j+1])*(0.1/1000)/2


            L2_deriv_avg.append(L2_deriv)
            L2_loss_avg.append(L2_loss)
            Vr_Loss_avg.append(vr_loss)

        L2_deriv_out = np.mean(L2_deriv_avg)
        L2_loss_out = np.mean(L2_loss_avg)
        Vr_loss_out = np.mean(Vr_Loss_avg)


        out_grad = [L2_deriv_out * g for g in grads] 

        return out_grad, [L2_loss_out,Vr_loss_out]
        



    elif grad_type == "vanRossum":


        #Set parameter 
        tau = 10 #(ms)
        #filter_length = 1000

        




        #TODO Bring in the target spikes
        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{matfile_path}/picture_fit.mat"
        data = scipy.io.loadmat(filename)

        #print(data)
        #print('here')
        #print(np.max(forwards_output,1))

        data = data['picture']

        repackaged_data = []
        #Repackage data
        #for k in range(len(data)):
        #    repackaged_data.append
        #    print(data[k][0])


        #print(data)

        #Build pictures
        loss = 0
        cumulative_deriv = 0

        #Loop through every trial
        for m in range(10):

            #print(forwards_output)
            #print('here')
            #print(np.shape(np.array(forwards_output)))
            #print(max(max(forwards_output)))

            vr_data = []

            vr_deriv = []

            #Go through the picture horizontally
            data_trial = data[m]
            sim_trial = np.array(forwards_output[m])

            

            data_spikes = np.where(data_trial == 1)[0]
            sim_spikes = np.where(sim_trial == 1)[0]


            #print(data_spikes)
            #print(sim_spikes)

            for k in range(np.shape(data)[1]):

                #print(k)
                



                #Find (f(t))
                data_val = 0
                data_deriv = 0
                for z in range(len(data_spikes)):
                    #print(data_spikes[z])
                    if data_spikes[z] <= k:
                        #Convert indicies and time constant to be in seconds
                        data_val += np.exp(-((k-data_spikes[z])/10000)/(tau/1000))
                        data_deriv += (1/(tau/1000))*np.exp(-((k-data_spikes[z])/10000)/(tau/1000))

                    #Accounts for the derivative of the heavyside
                    elif data_spikes[z] == k:
                        data_deriv -= np.exp(-((k-data_spikes[z])/10000)/(tau/1000))

                    else:
                        data_val += 0
                        data_deriv += 0


                #Find g(t)
                sim_val = 0
                sim_deriv = 0


                for z in range(len(sim_spikes)):
                    if sim_spikes[z] <= k:
                        #Convert indicies and time constant to be in seconds
                        sim_val += np.exp(-((k-sim_spikes[z])/10000)/(tau/1000))
                        sim_deriv += (1/(tau/1000))*np.exp(-((k-sim_spikes[z])/10000)/(tau/1000))

                    #Accounts for the derivative of the heavyside
                    elif sim_spikes[z] == k:
                        sim_deriv -= np.exp(-((k-sim_spikes[z])/10000)/(tau/1000))

                    else:
                        sim_val += 0
                        sim_deriv += 0

                vr_data.append((sim_val-data_val)**2)
                vr_deriv.append(2*(sim_deriv-data_deriv))
            

            #Take the auc of vr_data and vr_deriv
            for j in range(len(vr_data)-1):
                #0.1/1000 is dt in seconds (slice width)
                loss += (vr_data[j]+vr_data[j+1])*(0.1/1000)/2

                cumulative_deriv += (vr_deriv[j]+vr_deriv[j+1])*(0.1/1000)/2


        out_grad = [cumulative_deriv * g for g in grads] 

            






        #print('here')

        #print(np.shape(data))
        #print(np.shape(forwards_output),flush=True)

        #Reshape the data
        #output_reshaped = np.reshape(fowards_output,(1,int((29801*scale_factor-2)*10)))
        #data_reshpaed = '\TODO'

        return out_grad, loss




    else:
        print("please enter valid loss type")
        return
