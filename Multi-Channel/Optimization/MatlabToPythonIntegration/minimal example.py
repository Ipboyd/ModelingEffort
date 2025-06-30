import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
from torch.cuda.amp import autocast

#torch.autograd.set_detect_anomaly(True)

class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dt = 0.0001  # 0.1 ms
        self.tau = 0.02   # 20 ms
        self.V_th = 1.0
        self.V_reset = 0.0
        self.V_rest = 0.0
        self.input_current = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))  # Learnable input

    def forward(self):
        T = 350000  # 35s at 0.1ms resolution
        V = torch.zeros(T, dtype=torch.float32)
        spikes = torch.zeros(T, dtype=torch.float32)

        v_t = self.V_rest

        for t in range(1, T):
            dv = (-v_t + self.input_current) * (self.dt / self.tau)
            v_t = v_t + dv

            if v_t >= self.V_th:
                spikes[t] = 1.0
                v_t = self.V_reset

            V[t] = v_t

        return spikes

def main():
    model = LIF_ODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.0, 0.999))

    num_epochs = 1
    target_spikes = torch.tensor(50.0, dtype=torch.float32)  # target spike count

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        with autocast():
            output = model()  # forward pass

        fr = output.sum() / 10 / 3  # e.g., convert spike count to Hz

        loss = (fr - target_spikes) ** 2
        loss.backward()
        optimizer.step()

        gc.collect()

        print(f"Epoch {epoch}: Loss = {loss.item()}", flush=True)

if __name__ == "__main__":
    main()