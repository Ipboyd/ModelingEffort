
import numpy as np
from scipy.io import loadmat

def pinit(batch_size, num_params, name, f_path = None):

    if f_path != None:
        #print('here0')

        params = loadmat(f_path, squeeze_me=False, struct_as_record=False)["param_tracker"]
        losses = loadmat(f_path, squeeze_me=False, struct_as_record=False)["losses"]

        losses_PSTH = losses[:,1,:]

        min_idx = np.unravel_index(losses_PSTH.argmin(), losses_PSTH.shape)

        #print(np.unravel_index(losses_PSTH.argmin(), losses_PSTH.shape))
        #print(losses_PSTH[min_idx[0],min_idx[1]])

        p = params[min_idx[0],:,min_idx[1]][:,None]*np.ones((num_params,batch_size))

        #print(np.shape(p))

        return p

        



    else:

        p = np.zeros((num_params,batch_size))
        rng = np.random

        if name == "generated_Grad_Layer4":

            #print('here1')

            p[0:4,:] = rng.uniform(0.0, 0.08, size=(4, batch_size)).astype(np.float32) #GSYNs
            p[4:8,:] = rng.uniform(0.5, 15, size=(4, batch_size)).astype(np.float32) #T_refs
            p[8:10,:] = rng.uniform(0.05, 0.3, size=(2, batch_size)).astype(np.float32) #Input Gains

            #print(np.shape(p))

        elif name == "generated_Grad_Layer4v2":

            p[0:5,:] = rng.uniform(0.0, 0.08, size=(5, batch_size)).astype(np.float32) #GSYNs
            p[5:9,:] = rng.uniform(0.5, 15, size=(4, batch_size)).astype(np.float32) #T_refs
            p[9:11,:] = rng.uniform(0.05, 0.3, size=(2, batch_size)).astype(np.float32) #Input Gains

        elif name == "generated_Grad_Layer23":

            print('here2')

            p[0:10,:] = rng.uniform(0.0, 0.08, size=(10, batch_size)).astype(np.float32) #GSYNs
            p[10:17,:] = rng.uniform(0.5, 15, size=(7, batch_size)).astype(np.float32) #T_refs
            p[17:19,:] = rng.uniform(0.05, 0.3, size=(2, batch_size)).astype(np.float32) #Input Gains

        elif name == "generated_Grad_Layer56":

            print('here3')

            p[0:16,:] = rng.uniform(0.0, 0.08, size=(16, batch_size)).astype(np.float32) #GSYNs
            p[16:26,:] = rng.uniform(0.5, 15, size=(10, batch_size)).astype(np.float32) #T_refs
            p[26:28,:] = rng.uniform(0.05, 0.3, size=(2, batch_size)).astype(np.float32) #Input Gains

        else:
        
        
            #t_refs
            #p[0,:] = rng.uniform(0.5, 15, size=(1, batch_size)).astype(np.float32) #Neighborhood for Taus
            #p[4:12:2,:] = rng.uniform(0.0, 0.08, size=(4, batch_size)).astype(np.float32) #Neighborhood for Gsyns
            #p[5:13:2,:] = rng.uniform(0.1, 0.9, size=(4, batch_size)).astype(np.float32) #Neighborhood for Fps
            #p[12,:] = rng.uniform(1, 15, size=(1, batch_size)).astype(np.float32) #Neighborhood for FR

            #p[0:4,:] = np.ones((4,400)) * np.arange(0.5,15,(15-0.5)/400) #Init Taus to be the same accross the board
            #p[0,:] = rng.uniform(0.05,0.17, size=(1, batch_size)).astype(np.float32)
            p[0,:] = np.ones((1,400)) * np.arange(0.05,0.25,(0.25-0.05)/400) #Grid search Input Gains


    return p

