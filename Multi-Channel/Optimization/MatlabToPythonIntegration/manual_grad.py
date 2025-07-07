import numpy as np
import time
import matplotlib.pyplot as plt


def forwards(p, T, thresh, c):

    #Preallocated arrays (to be used in FR or PSTH)
    V = np.zeros(T,)
    spikes = np.zeros(T,)

    #Zero V_t to start
    v_t = 0.0

    #Solve the "ODEs" with the Eulers method
    for t in range(1, T):
        dv = (-v_t + p) * c
        v_t = v_t + dv

        if v_t >= thresh:
            spikes[t] = 1.0
            v_t = 0.0

        V[t] = v_t

    return spikes, V


def backwards(fr, target, V, T, thresh, c):

    grad = 0

    for t in range(1, T):
        grad += ((6*np.exp(-6*(V[t] - thresh - 0.5)))/(1+np.exp(-6*(V[t] - thresh - 0.5)))**2)*c
        print(grad)

    grad = grad*2*(fr-target)

    return grad

start = time.time()

T = 3500000 
thresh = 1.0
dt = 0.0001
tau = 0.02
c = dt/tau

#training loop

epochs = 100
p = 1.5   #Initial parameter value

#Adam
m = 0
v = 0
beta1, beta2 = 0.7, 0.997
eps = 1e-8
t = 0
lr = 3e-1

target = 200
losses = []

for k in range(epochs):

    out_spikes, out_voltage = forwards(p,T,thresh,c)
    
    fr = sum(out_spikes)/10/3
    loss = (fr - target)**2
    losses.append(loss)

    print(fr)

    print(loss)

    out_grad = backwards(fr, target, out_voltage, T, thresh, c)

    print(out_grad)

    t += 1
    m = beta1 * m + (1 - beta1) * out_grad
    v = beta2 * v + (1 - beta2) * (out_grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)

    #p = p - out_grad*lr

    print(p)

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

# Plot the loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(losses, marker='o')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()