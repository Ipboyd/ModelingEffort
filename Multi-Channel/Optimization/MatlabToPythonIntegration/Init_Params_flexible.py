
import numpy as np
from scipy.io import loadmat

def pinit(batch_size, num_params, name, load_from_file):

    if load_from_file:
        #file = f_path
        #mat = loadmat(file, squeeze_me=False, struct_as_record=False)["param_tracker"]
        #p = mat[-1,:,:]
        return

        



    else:

        p = np.zeros((num_params,batch_size))
        rng = np.random

        if name == "generated_Grad_Layer4":

            p[0:4,:] = rng.uniform(0.0, 0.08, size=(4, batch_size)).astype(np.float32) #GSYNs
            p[4:8,:] = rng.uniform(0.5, 15, size=(4, batch_size)).astype(np.float32) #T_refs
            

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

