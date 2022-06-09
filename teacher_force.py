from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from helpers import data2fn

save_dir_data = '/cnl/data/spate/Res/model_data/'
fcn = 'zero'
do_save = True
do_plot = False
d = 1
n_inputs = d
n_hidden = 2**10
n_outputs = d
spectral_radius = 1.2
pcon = 10 / n_hidden # Each unit connected to 10 other on average
gpu = -1 # -1 => cpu, 0 and above => gpu number
n_steps = 4000 # Number timesteps to simulate
n_samples = 100
sim_samples = 10
L = 10
leak_rate = 0.01 # Found 2 * frequency of the fcn is good

cluster_size = 2**0

res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu, cluster_size=cluster_size) # Create reservoir

# Get target trace
t = np.arange(n_steps)

if fcn == 'zero':
    target = np.zeros((n_samples, d, n_steps))
else:
    target_fn = data2fn[fcn]
    target = np.loadtxt(target_fn, delimiter=',').reshape(n_samples, d, n_steps)

target = target[:sim_samples, :, :]

# Simulate
states, outputs = res.simulate(n_steps, sim_samples, target=target)

# Plot simulation
ds = 50 # Downsample hidden units
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

if do_plot:
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax[0].plot(t, outputs[:,0,:].T)
    ax[0].plot(t, target[0,0,:], "k--")
    ax[1].plot(t, states[:,::ds,:].reshape(-1, n_steps).T)

    ax[0].set_title("Output")
    ax[1].set_title("Select states")
    fig.tight_layout()
    plt.show()

if do_save:
    np.savetxt(save_dir_data + f"teacher_force_states_{fcn}_n_units_{n_hidden}_sr_{spectral_radius}.csv", states.reshape(-1, n_steps), delimiter=',')