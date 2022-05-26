from esn import esn
import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers import data2fn

# General settings
sweep = 'average' # Type of params to search over: 'size', 'lambda', 'random_units', 'average'
n_replicates = 20
save_dir_fig = '/home/spate/Res/figures/'
save_dir_data = '/cnl/data/spate/Res/'
gpu = 0 # -1 => cpu, 0 and above => gpu number
do_norm = True # Normalize target
do_save = True

# Data
fcn = 'lorenz'
L = 10 # Period of sine
d = 3 # Dimensionality of system
total_samples = 500 # Total avail system time series
total_steps = 4000 # System simulated this many time steps
dt = 0.01

# Model
n_inputs = d
n_hidden = 2**10
n_outputs = d
spectral_radius = 1.2
leak_rate = 0.01

# Training
n_steps = total_steps # Number timesteps to train on
n_samples = 490 # Size of training batch
teacher_steps = 2000 # Number timesteps to teacher force during test
extend = 2000 # Timesteps to extend past teacher forcing during test
test_sample = 497
reg_method = 'pinv' # Regression method: 'ridge', 'lasso', 'pinv'
lam = 0 # Regression regularization param for ridge and lasso
ep = 1e-4 # Threshold under which regression coefficient is considered 0

# -------------------------------------------------------------------
# Params to search - overwrites above settings
net_sizes = 2**np.arange(1, 11)
lams = np.logspace(0, -5, base=10, num=6)
fracs = 2**np.arange(1, 10) / n_hidden
cluster_sizes = 2**np.arange(0, 10)
# -------------------------------------------------------------------

target_fn = data2fn[fcn] # Data filename from helpers.py

# Get data
t = np.arange(n_steps)
target = np.loadtxt(target_fn, delimiter=',').reshape(total_samples, d, total_steps)

# Normalize target data
if do_norm:
    flat_target = np.transpose(target, axes=[1, 0, 2]).reshape(n_outputs, -1)
    target_mean, target_std = np.mean(flat_target, axis=1), np.std(flat_target, axis=1)
    target -= target_mean.reshape(1, -1, 1)
    target /= target_std.reshape(1, -1, 1)

train_target = target[:n_samples, :n_outputs, :n_steps].reshape(n_samples, n_outputs, n_steps)
test_target = target[test_sample, :, :teacher_steps].reshape(1, n_outputs, teacher_steps) # Slice off beyond teacher steps

save_rmse = []
save_last_t = []

if sweep == 'size':
    for i, n_hidden in enumerate(net_sizes):
        print(f"Hidden units: {n_hidden}")
        
        rmse_arr = []
        last_t_arr = []
        for j in range(n_replicates):
            
            print(f"Replicate # {j}")
            # Create new reservoir
            pcon = 10 / n_hidden # Each unit connected to 10 other on average
            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

            # Tune w_out
            w_out_hat = res.fit(train_target, method=reg_method, lam=lam)
            res.set_w_out(w_out_hat) # Set w_out

            # Test
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            rmse = np.sqrt(np.mean(np.square(outputs - target_plot), axis=1)).reshape(-1,)
            rmse_truncated = rmse[1000:] # Cut off first 10 timesteps
            
            # Find last timestep of accurate prediction
            error_mask = rmse[teacher_steps:] > 1
            if np.all(error_mask): # Error above thresh at all times
                last_t = 0
            elif not np.any(error_mask): # Error stays below thresh
                last_t = extend * dt
            else:
                last_t = (np.where(error_mask)[0][0] + 1) * dt
            
            rmse_arr.append(rmse_truncated.mean())
            last_t_arr.append(last_t)
        
        save_last_t.append([n_hidden, np.array(last_t_arr).mean(), np.array(last_t_arr).std()])
        save_rmse.append([n_hidden, np.array(rmse_arr).mean(), np.array(rmse_arr).std()])

elif sweep == 'lambda':
    for i, lam in enumerate(lams):
        print(f"Lambda: {lam}")
        
        rmse_arr = []
        last_t_arr = []
        nonzero_frac_arr = []
        for j in range(n_replicates):
            
            print(f"Replicate # {j}")
            # Create new reservoir
            pcon = 10 / n_hidden # Each unit connected to 10 other on average
            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

            # Tune w_out
            w_out_hat = res.fit(train_target, method=reg_method, lam=lam)
            res.set_w_out(w_out_hat) # Set w_out

            # Test
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            rmse = np.sqrt(np.mean(np.square(outputs - target_plot), axis=1)).reshape(-1,) 
            rmse_truncated = rmse[1000:] # Cut off first 10 timesteps

            nonzero_frac = w_out_hat[np.abs(w_out_hat) > ep].size / w_out_hat.size
            
            # Find last timestep of accurate prediction
            error_mask = rmse[teacher_steps:] > 1
            if np.all(error_mask): # Error above thresh at all times
                last_t = 0
            elif not np.any(error_mask): # Error stays below thresh
                last_t = extend * dt
            else:
                last_t = (np.where(error_mask)[0][0] + 1) * dt
            
            rmse_arr.append(rmse_truncated.mean())
            last_t_arr.append(last_t)
            nonzero_frac_arr.append(nonzero_frac)
        
        save_last_t.append([lam, np.array(last_t_arr).mean(), np.array(last_t_arr).std(), np.array(nonzero_frac_arr).mean(), np.array(nonzero_frac_arr).std()])
        save_rmse.append([lam, np.array(rmse_arr).mean(), np.array(rmse_arr).std()])

elif sweep == 'random_units':
    for i, frac in enumerate(fracs):
        print(f"Frac: {frac}")
        
        rmse_arr = []
        last_t_arr = []
        for j in range(n_replicates):
            
            print(f"Replicate # {j}")
            # Create new reservoir
            pcon = 10 / n_hidden # Each unit connected to 10 other on average
            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu) # Create reservoir

            # Tune w_out
            w_out_hat = res.fit_rand_units(train_target, frac=frac)
            res.set_w_out(w_out_hat) # Set w_out

            # Test
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            rmse = np.sqrt(np.mean(np.square(outputs - target_plot), axis=1)).reshape(-1,) 
            rmse_truncated = rmse[1000:] # Cut off first 10 timesteps
            
            # Find last timestep of accurate prediction
            error_mask = rmse[teacher_steps:] > 1
            if np.all(error_mask): # Error above thresh at all times
                last_t = 0
            elif not np.any(error_mask): # Error stays below thresh
                last_t = extend * dt
            else:
                last_t = (np.where(error_mask)[0][0] + 1) * dt
            
            rmse_arr.append(rmse_truncated.mean())
            last_t_arr.append(last_t)
        
        save_last_t.append([frac, np.array(last_t_arr).mean(), np.array(last_t_arr).std()])
        save_rmse.append([frac, np.array(rmse_arr).mean(), np.array(rmse_arr).std()])

elif sweep == 'average':
    for i, cluster_size in enumerate(cluster_sizes):
        print(f"Cluster size: {cluster_size}")
        
        rmse_arr = []
        last_t_arr = []
        for j in range(n_replicates):
            
            print(f"Replicate # {j}")
            # Create new reservoir
            pcon = 10 / n_hidden # Each unit connected to 10 other on average
            res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu, cluster_size=cluster_size) # Create reservoir

            # Tune w_out
            w_out_hat = res.fit(train_target, method=reg_method, lam=lam)
            res.set_w_out(w_out_hat) # Set w_out

            # Test
            states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
            states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

            # Compute error
            target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
            rmse = np.sqrt(np.mean(np.square(outputs - target_plot), axis=1)).reshape(-1,) 
            rmse_truncated = rmse[1000:] # Cut off first 10 timesteps
            
            # Find last timestep of accurate prediction
            error_mask = rmse[teacher_steps:] > 1
            if np.all(error_mask): # Error above thresh at all times
                last_t = 0
            elif not np.any(error_mask): # Error stays below thresh
                last_t = extend * dt
            else:
                last_t = (np.where(error_mask)[0][0] + 1) * dt
            
            rmse_arr.append(rmse_truncated.mean())
            last_t_arr.append(last_t)
        
        save_last_t.append([cluster_size, np.array(last_t_arr).mean(), np.array(last_t_arr).std()])
        save_rmse.append([cluster_size, np.array(rmse_arr).mean(), np.array(rmse_arr).std()])

if do_save:
    save_last_t = np.array(save_last_t)
    save_rmse = np.array(save_rmse)
    if sweep == 'size':
        if fcn == 'sine':
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_period_{L}_{sweep}_last_t_predicted_sr_{spectral_radius}_lr_{leak_rate}.csv", save_last_t, delimiter=',')
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_period_{L}_{sweep}_rmse_sr_{spectral_radius}_lr_{leak_rate}.csv", save_rmse, delimiter=',')
        else:
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_{sweep}_last_t_predicted_sr_{spectral_radius}_lr_{leak_rate}.csv", save_last_t, delimiter=',')
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_{sweep}_rmse_sr_{spectral_radius}_lr_{leak_rate}.csv", save_rmse, delimiter=',')
    else:
        if fcn == 'sine':
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_period_{L}_{sweep}_last_t_predicted_n_hidden_{n_hidden}_sr_{spectral_radius}_lr_{leak_rate}.csv", save_last_t, delimiter=',')
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_period_{L}_{sweep}_rmse_n_hidden_{n_hidden}_sr_{spectral_radius}_lr_{leak_rate}.csv", save_rmse, delimiter=',')
        else:
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_{sweep}_last_t_predicted_n_hidden_{n_hidden}_sr_{spectral_radius}_lr_{leak_rate}.csv", save_last_t, delimiter=',')
            np.savetxt(f"/cnl/data/spate/Res/{fcn}_{sweep}_rmse_n_hidden_{n_hidden}_sr_{spectral_radius}_lr_{leak_rate}.csv", save_rmse, delimiter=',')