from esn import esn
import numpy as np
import torch
import matplotlib.pyplot as plt

# General settings
save_dir_fig = '/home/spate/Res/figures/'
save_dir_data = '/cnl/data/spate/Res/'
gpu = -1 # -1 => cpu, 0 and above => gpu number
do_norm = True # Normalize target
do_plot = True
do_save = False

# Data
fcn = 'sine'
L = 10 # Period of sine
d = 1 # Dimensionality of system
total_samples = 100 # Total avail system time series
total_steps = 4000 # System simulated this many time steps

# Model
n_inputs = 1
n_hidden = 2
n_outputs = 1
spectral_radius = 0.8
leak_rate = 0.002
pcon = 10 / n_hidden # Each unit connected to 10 other on average

# Training
n_steps = total_steps # Number timesteps to train on
n_samples = total_samples # Size of training batch
teacher_steps = 2000 # Number timesteps to teacher force during test
extend = 2000 # Timesteps to extend past teacher forcing during test
# test_sample = np.random.randint(n_samples, total_samples) # The single target to test on multiple times
test_sample = 0
reg_method = 'pinv' # Regression method: 'ridge', 'lasso', 'pinv'
lam = 0 # Regression regularization param for ridge and lasso

# Data file
if fcn == 'lorenz':
    target_fn = f"/home/spate/Res/targets/lorenz_params_sig_10.00_rho_28.00_beta_2.67_n_samples_{total_samples}_n_steps_{total_steps}_dt_0.01.csv"
elif fcn == 'rossler':
    target_fn = f"/home/spate/Res/targets/rossler_params_a_0.20_b_0.20_c_5.70_n_samples_{total_samples}_n_steps_{total_steps}_dt_0.01.csv"
elif fcn == 'sine':
    target_fn = f"/home/spate/Res/targets/sine_period_{L}_n_samples_{total_samples}_n_steps_{total_steps}_dt_0.01.csv"