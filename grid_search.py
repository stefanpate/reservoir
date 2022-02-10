from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

target_fn = '/home/spate/Res/targets/lorenz_params_sig_10.00_rho_28.00_beta_2.67_n_samples_500_n_steps_4000_dt_0.01.csv'
n_inputs = 1
n_hidden = 1000
n_outputs = 3
d = 3 # Dimensionality of system
total_samples = 500 # Total avail system time series
total_steps = 4000 # System simulated this many time steps
pcon = 10 / n_hidden # Each unit connected to 10 other on average
gpu = 3 # -1 => cpu, 0 and above => gpu number
n_steps = 3000 # Number timesteps to simulate
n_samples = 490 # In training batch
teacher_steps = 2000 # Number timesteps to teacher force during test
extend = 100 # Timesteps to extend past teacher forcing during test
do_norm = True

# Grid search params
n_repeats = 10
leak_rates = np.array([1, 0.5, 0.1, 0.05, 0.02, 0.01])
spectral_radii = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
mse = np.zeros(shape=(spectral_radii.shape[0], leak_rates.shape[0], n_repeats)) - 1

# Grid search
for i, spectral_radius in enumerate(spectral_radii):
    for j, leak_rate in enumerate(leak_rates):

        print(f"Computing MSE for SR: {spectral_radius}, LR: {leak_rate}.")
        for k in range(n_repeats):

            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

            # Get target trace
            t = np.arange(n_steps)
            target = np.loadtxt(target_fn, delimiter=',').reshape(total_samples, d, total_steps)

            # Normalize target data
            if do_norm:
                flat_target = np.transpose(target, axes=[1, 0, 2]).reshape(n_outputs, -1)
                target_mean, target_std = np.mean(flat_target, axis=1), np.std(flat_target, axis=1)
                target -= target_mean.reshape(1, -1, 1)
                target /= target_std.reshape(1, -1, 1)

            train_target = target[:n_samples, :n_outputs, :n_steps].reshape(n_samples, n_outputs, n_steps) # x-coordinate of target system

            # Tune w_out
            w_out_hat_torch = res.fit(train_target)
            w_out_hat = w_out_hat_torch.cpu().numpy()
            w_out = res.w_out.cpu().numpy()
            res.set_w_out(w_out_hat_torch) # Set w_out

            # Test
            test_sample = np.random.randint(n_samples, total_samples) # The single target to test on multiple times
            test_target = target[test_sample, :, :teacher_steps].reshape(1, n_outputs, teacher_steps) # Slice off beyond teacher steps
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute mean square error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            this_mse = np.mean(np.square(outputs - target_plot), axis=1).reshape(-1,) # Average over dim of syst
            this_mse = np.mean(this_mse[teacher_steps: teacher_steps + extend]) # Average over time

            mse[i,j,k] = this_mse

mse = mse.reshape(spectral_radii.shape[0], -1)
np.savetxt(f"/home/spate/Res/lorenz_grid_search_spectral_radius_leak_rate_hidden_units_{n_hidden}.csv", mse, delimiter=',')
            