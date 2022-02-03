from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

n_inputs = 1
n_hidden = 500
n_outputs = 1
spectral_radius = 0.8
pcon = 10 / n_hidden # Each unit connected to 10 other on average
gpu = -1 # -1 => cpu, 0 and above => gpu number
n_steps = 3000 # Number timesteps to simulate
L = 1000 # Period of target function
leak_rate = 2 / L # Found 2 * frequency of the fcn is good
n_samples = 100 # In training batch
test_samples = 10
teacher_steps = 2000 # Number timesteps to teacher force during test
extend = 2000 # Timesteps to extend past teacher forcing during test
tau = 0 # Delay width of target relative to input
fcn = lambda x : np.sin(2 * np.pi * (1 / L) * x) # Fcn to predict

res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

# Make target trace
t = np.arange(0, n_steps)
target = [fcn(t - tau).reshape(n_outputs, n_steps),] * n_samples
target = np.stack(target, axis=0)

# Tune w_out
w_out_hat_torch = res.fit(target)
w_out_hat = w_out_hat_torch.cpu().numpy()
w_out = res.w_out.cpu().numpy()

# Test
res.set_w_out(w_out_hat_torch) # Set w_out
test_target = target[:test_samples,:,:teacher_steps] # Slice off beyond teacher steps
states, outputs = res.simulate(teacher_steps + extend, test_samples, target=test_target, extend=extend)

# Plot after tuning w_out
ds = 50 # Downsample hidden units to plot
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()
t_plot = np.arange(0, teacher_steps + extend)
# target_plot = target[0,0,:teacher_steps + extend]
target_plot = fcn(t_plot - tau)

fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
ax[0].plot(t_plot, outputs[:,0,:].T)
ax[0].plot(t_plot, target_plot, "k--")
ax[1].plot(t_plot, states[:,::ds,:].reshape(-1, len(t_plot)).T)

ax[0].set_title("Output")
ax[1].set_title("Select states")
fig.tight_layout()
plt.show()