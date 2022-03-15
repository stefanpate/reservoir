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
n_hidden = 10
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
# w_out_hat = res.fit(train_target, method=reg_method, lam=lam)
w_out_hat = res.fit_rand_units(train_target, frac=0.5)
w_out = res.w_out.cpu().numpy()
res.set_w_out(w_out_hat) # Set w_out

# Test
test_target = target[test_sample, :, :teacher_steps].reshape(1, n_outputs, teacher_steps) # Slice off beyond teacher steps
states, outputs = res.simulate(teacher_steps + extend, 1, target=test_target, extend=extend)
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

# Compute mean square error
target_plot = target[test_sample,:,:teacher_steps + extend].reshape(1, n_outputs, teacher_steps + extend)
square_error = np.mean(np.square(outputs - target_plot), axis=1).reshape(-1,)

# Plot after tuning w_out
if do_plot:
    ds = 30 # Downsample hidden units to plot
    var_names = ["x", "y", "z"]
    t_plot = np.arange(0, teacher_steps + extend)

    print(f"Test sample: {test_sample}")
    print(f"MSE TF: {np.mean(square_error[:teacher_steps]):.2f}, MSE OFB: {np.mean(square_error[teacher_steps:]):.2f}")
    gs = [1] * n_outputs + [0.5]
    fig, ax = plt.subplots(n_outputs + 1, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios':gs})
    
    for i in range(n_outputs):
        ax[i].plot(t_plot * 0.01, outputs[:,i,:].T)
        ax[i].plot(t_plot * 0.01, target_plot[0,i,:], "k--")
        ax[i].set_ylabel(var_names[i])
    ax[-1].plot(t_plot * 0.01, states[:,::ds,:].reshape(-1, len(t_plot)).T)

    if not do_norm:
        ax[0].set_ylim(outputs[:,0,:].min(), outputs[:,0,:].max())
    
    ax[0].set_title("Output")
    ax[-1].set_title("Select states")
    ax[-1].set_xlabel("Time")
    ax[-1].set_ylabel("Hidden unit activity")
    fig.tight_layout()
    plt.savefig(save_dir_fig + f"{fcn}_prediction_test_sample_{test_sample}_output_target_states_n_{n_hidden}_sr_{spectral_radius}_lr_{leak_rate}.png")
    plt.show()

# Save
if do_save:
    np.savetxt(save_dir_data + f"{fcn}_prediction_target_test_sample_{test_sample}.csv", target_plot.reshape(n_outputs, teacher_steps + extend), delimiter=',')
    np.savetxt(save_dir_data + f"{fcn}_prediction_outputs_test_sample_{test_sample}.csv", outputs.reshape(n_outputs, teacher_steps + extend), delimiter=',')