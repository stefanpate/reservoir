from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

target_fn = '/home/spate/Res/targets/sine_period_10_n_samples_100_n_steps_4000_dt_0.01.csv'
n_inputs = 1
n_hidden = 2**6
n_outputs = 1
spectral_radius = 0.8
pcon = 10 / n_hidden # Each unit connected to 10 other on average
gpu = -1 # -1 => cpu, 0 and above => gpu number
n_steps = 4000 # Number timesteps to simulate
n_samples = 100
L = 10
leak_rate = 2 / L # Found 2 * frequency of the fcn is good
d = 1
cluster_size = 2**6

res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu, cluster_size=cluster_size) # Create reservoir

# Get target trace
t = np.arange(n_steps)
target = np.loadtxt(target_fn, delimiter=',').reshape(n_samples, d, n_steps)
# target = sol[0,0,:n_steps].reshape(n_samples, n_outputs, n_steps) # x-coordinate of target system

# Simulate
states, outputs = res.simulate(n_steps, n_samples, target=target)

# Plot simulation
ds = 50 # Downsample hidden units
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ax[0].plot(t, outputs[:,0,:].T)
ax[0].plot(t, target[0,0,:], "k--")
ax[1].plot(t, states[:,::ds,:].reshape(-1, n_steps).T)

ax[0].set_title("Output")
ax[1].set_title("Select states")
fig.tight_layout()
plt.show()