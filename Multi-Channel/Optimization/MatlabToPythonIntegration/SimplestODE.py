import torch
from torch import nn
from torchdiffeq import odeint

class SimplestODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))  # learnable parameter

    def forward(self, t, y):
        return self.a * y  # dy/dt = a*y

# Setup
y0 = torch.tensor([1.0])  # initial value
t = torch.tensor([0.0, 5.0])  # integrate from 0 to 5
target = torch.tensor([170.0])

model = SimplestODE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
for step in range(500):
    optimizer.zero_grad()
    pred_y = odeint(model, y0, t)
    loss = (pred_y[-1] - target).pow(2).mean()
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == 499:
        print(f"Step {step:03d} | Loss: {loss.item():.4f} | a: {model.a.item():.4f}")