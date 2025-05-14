import torch
from torch import nn
from torchdiffeq import odeint

class CoupledNonlinearODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, t, z):
        x, y = z[0], z[1]
        dxdt = self.a * x
        dydt = self.b * x * y  # nonlinear coupling
        return torch.stack([dxdt, dydt])

# Initial conditions
z0 = torch.tensor([1.0, 1.0])  # x(0)=1, y(0)=1
t = torch.linspace(0, 5, 100)

model = CoupledNonlinearODE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

target_y_at_t5 = torch.tensor(170.0)

# Training loop
for step in range(500):
    optimizer.zero_grad()
    pred_z = odeint(model, z0, t)
    pred_y = pred_z[-1, 1]  # y at t=5
    loss = (pred_y - target_y_at_t5).pow(2)
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == 499:
        print(f"Step {step:03d} | Loss: {loss.item():.4f} | a: {model.a.item():.4f}, b: {model.b.item():.4f}")

import matplotlib.pyplot as plt
plt.plot(t.numpy(), pred_z[:,1].detach().numpy())
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('dy/dt = theta * y')
plt.grid(True)
plt.show()


#WARNING. NEED TO BE ABLE TO CONTROL GRANULARITY
#This requires actually setting things in odeint
#Right now it defaults to dopri5 but we could do odeint(..., options{'step_size' : 0.01})

