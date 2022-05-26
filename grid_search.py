from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import sys

'''
Command line instructions:

1. Hidden units
2. Gpu. -1 => cpu, 0 and above => gpu number

Example command:
python grid_search.py 100 0
'''

# General settings
data_dir = '/cnl/data/spate/Datasets/'
save_dir = '/cnl/data/spate/Res/'
do_norm = True
gpu = int(sys.argv[2])

# Data
fcn = 'mackey_glass'
d = 1 # Dimensionality of system
total_samples = 500 # Total avail system time series
total_steps = 4000 # System simulated this many time steps
L = 10 # Period of sine
dt = 0.01

# Model
n_hidden = int(sys.argv[1])
n_inputs = d
n_outputs = d
pcon = 10 / n_hidden # Each unit connected to 10 other on average

# Training
n_steps = total_steps # Number timesteps to simulate in training
n_samples = 490 # In training batch
teacher_steps = 2000 # Number timesteps to teacher force during test
extend = 2000 # Timesteps to extend past teacher forcing during test

reg_method = 'pinv' # Regression method: 'ridge', 'lasso', 'pinv'
lam = 0 # Regression regularization param for ridge and lasso
ep = 1e-4 # Threshold under which regression coefficient is considered 0

# Grid search params
n_repeats = 10
leak_rates = np.array([1, 0.5, 0.1, 0.05, 0.02, 0.01])
spectral_radii = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

# Get data file
if fcn == 'lorenz':
    target_fn = f"{data_dir}lorenz_params_sig_10.00_rho_28.00_beta_2.67_n_samples_{total_samples}_n_steps_{total_steps}_dt_{dt}.csv"
elif fcn == 'rossler':
    target_fn = f"{data_dir}rossler_params_a_0.20_b_0.20_c_5.70_n_samples_{total_samples}_n_steps_{total_steps}_dt_{dt}.csv"
elif fcn == 'sine':
    target_fn = f"{data_dir}sine_period_{L}_n_samples_{total_samples}_n_steps_{total_steps}_dt_{dt}.csv"
elif fcn == 'mackey_glass':
    target_fn = f"{data_dir}mackey_glass_beta_2_gamma_1_n_9.65_tau_2_n_samples_{total_samples}_n_steps_{total_steps}_dt_{dt}.csv"

# Get data
t = np.arange(n_steps)
target = np.loadtxt(target_fn, delimiter=',').reshape(total_samples, d, total_steps) # All data
train_target = target[:n_samples, :n_outputs, :n_steps].reshape(n_samples, n_outputs, n_steps) # Training set

# Normalize target data
if do_norm:
    flat_target = np.transpose(target, axes=[1, 0, 2]).reshape(n_outputs, -1)
    target_mean, target_std = np.mean(flat_target, axis=1), np.std(flat_target, axis=1)
    target -= target_mean.reshape(1, -1, 1)
    target /= target_std.reshape(1, -1, 1)

# Grid search
save_last_t = np.zeros(shape=(spectral_radii.shape[0], leak_rates.shape[0], n_repeats))
for i, spectral_radius in enumerate(spectral_radii):
    for j, leak_rate in enumerate(leak_rates):

        print(f"Computing performance for Spectral radius: {spectral_radius}, Leak rate: {leak_rate}.")
        for k in range(n_repeats):

            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

            # Tune w_out
            w_out_hat_torch = res.fit(train_target, method=reg_method, lam=lam)
            res.set_w_out(w_out_hat_torch) # Set w_out

            # Test
            test_sample = np.random.randint(n_samples, total_samples) # The single target to test on multiple times
            test_target = target[test_sample, :, :teacher_steps].reshape(1, n_outputs, teacher_steps) # Slice off beyond teacher steps
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute root mean square error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            rmse = np.sqrt(np.mean(np.square(outputs - target_plot), axis=1)).reshape(-1,)

            # Find last timestep of accurate prediction
            error_mask = rmse[teacher_steps:] > 1
            if np.all(error_mask): # Error above thresh at all times
                last_t = 0
            elif not np.any(error_mask): # Error stays below thresh
                last_t = extend * dt
            else:
                last_t = (np.where(error_mask)[0][0] + 1) * dt


            save_last_t[i,j,k] = last_t

mse = save_last_t.reshape(spectral_radii.shape[0], -1)
np.savetxt(f"{save_dir}{fcn}_grid_search_spectral_radius_leak_rate_hidden_units_{n_hidden}.csv", mse, delimiter=',')