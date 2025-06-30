import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
from torch.cuda.amp import autocast
import time





class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()


        #There here are just some example neuronal parameters (not necessarily biologically relavent/realistic)

        self.dt = 0.0001  # 0.1 ms
        self.tau = 0.02   # 20 ms
        self.V_th = 1.0
        self.V_reset = 0.0
        self.V_rest = 0.0

        #Here we are trying to learn the level of input current that creates a certain amount of spiking

        self.input_current = nn.Parameter(torch.tensor(1.5, dtype=torch.float32)) 

    def forward(self):

        #Here we do 3.5 million timesteps. In the other simulation I do 35k timsteps 10 times with 8 different neuron populations (so this might be a good proxy for the
        #size of graphs that I am building)

        T = 3500000 
        #T = 35000 
        V = torch.zeros(T, dtype=torch.float32)
        spikes = torch.zeros(T, dtype=torch.float32)

        v_t = self.V_rest

        #Very simple spiking activity

        for t in range(1, T):
            dv = (-v_t + self.input_current) * (self.dt / self.tau)
            v_t = v_t + dv

            if v_t >= self.V_th:
                spikes[t] = 1.0
                v_t = self.V_reset

            V[t] = v_t

        return spikes

def main():
    print('made it here')
    start = time.time()
    model = LIF_ODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.0, 0.999))

    num_epochs = 1
    target_spikes = torch.tensor(50.0, dtype=torch.float32)  # target spike count

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        #Reduces the storage size by up to half here by using autocast
        #with autocast():
        output = model()  # forward pass
        
        print('made it here')
        
        #Take loss and backpropegate
        
        fr = output.sum() / 10 / 3  

        loss = (fr - target_spikes) ** 2
        loss.backward()
        optimizer.step()

        gc.collect()

        print(f"Epoch {epoch}: Loss = {loss.item()}", flush=True)

    # your code here
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")

if __name__ == "__main__":
    print('here')
    main()
    print('here')
    print("Done timing.", flush=True)