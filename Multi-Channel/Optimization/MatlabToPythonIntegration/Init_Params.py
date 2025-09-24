import numpy as np
from scipy.io import loadmat

def pinit(batch_size, num_params,learning_mask,load_from_file):

    if load_from_file:
        file = "results//run_2025-08-12_15-22-22.mat"
        mat = loadmat(file, squeeze_me=False, struct_as_record=False)["param_tracker"]
        p = mat[-1,:,:]

        #print(p)
    else:
        #Random initialization bounded by 0 and 0.08 uS
        p = np.zeros((num_params,batch_size))
        rng = np.random
        p[0:10,:] = rng.uniform(0.0, 0.08, size=(10, batch_size)).astype(np.float32) #Neighborhood for Gsyns

        if learning_mask[1] == 1:
            p[10:17,:] = rng.uniform(5, 25, size=(7, batch_size)).astype(np.float32) #Neighborhood for Taus

        #print(p)

    return p
