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
n_steps = 3000 # Timesteps of simulation
n_periods = n_steps / 1000
leak_rate = 10 / n_steps # Found 10 * delta_t is pretty good
n_samples = 100
pass_target = True
input_fcns = [np.sin,] * n_samples
tau = (n_periods * 2 * np.pi) / n_steps # Delay width of target relative to input
test_n_samples = 10
test_n_steps = 3000
test_n_periods = test_n_steps / 1000

# Check I can instantiate
res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu)
# print(res)

# Check SR is same as above
# evals, _ = torch.eig(res.w)
# print("Spectral radius: ", (evals**2).sum(dim=1).sqrt_().max())

# Check w pcon is correct
# n_zero_weights = (res.w == 0).to(dtype=torch.float).sum()
# print("Cxn density: ", (n_hidden**2 - n_zero_weights) / n_hidden**2)

# Create n_samples samples of input
t = np.linspace(0, n_periods * 2 * np.pi, num=n_steps)
input = []
for f in input_fcns:
    this_input = f(t).reshape(n_inputs, n_steps)
    input.append(this_input)
input = np.stack(input, axis=0) # (3, n_inputs, n_steps)

# Create n_samples of target
target = []
for f in input_fcns:
    this_target = f(t - tau).reshape(n_outputs, n_steps)
    target.append(this_target)
target = np.stack(target, axis=0)

# Simulate. Measure compute time
start = time.time()

if pass_target:
    states, outputs = res.simulate(n_steps, n_samples, target=target)
else:
    states, outputs = res.simulate(n_steps, n_samples)

end = time.time()
elapsed = end - start

if gpu == -1:
    device = 'CPU'
else:
    device = torch.cuda.get_device_name(gpu)

print(f"Device: {device}, Time elapsed = {elapsed:.4f} seconds")


# Plot simulation
ds = 50 # Downsample
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
ax[0].plot(t, input[:,0,:].T)
ax[1].plot(t, outputs[:,0,:].T)
ax[1].plot(t, target[0,0,:], "k--")
ax[2].plot(t, states[:,::ds,:].reshape(-1, n_steps).T)

ax[0].set_title("Input")
ax[1].set_title("Output")
ax[2].set_title("Select states")
fig.tight_layout()
plt.show()

# Tune w_out
w_out_hat_torch = res.fit(target)
w_out_hat = w_out_hat_torch.cpu().numpy()
w_out = res.w_out.cpu().numpy()

# Plot pre/post output matrices
# plt.hist(w_out.reshape(-1))
# plt.hist(w_out_hat.reshape(-1), alpha=0.5)
# plt.show()

res.set_w_out(w_out_hat_torch) # Set w_out
test_t = np.linspace(0, test_n_periods * 2 * np.pi, num=test_n_steps)
test_target = [np.sin(test_t - tau).reshape(n_outputs, test_n_steps),] * test_n_samples
test_target = np.stack(test_target, axis=0)


states, outputs = res.simulate(test_n_steps, test_n_samples, target=test_target) # Simulate again

# Plot after tuning w_out
ds = 50 # Downsample hidden units to plot
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
ax[0].plot(test_t, outputs[:,0,:].T)
ax[0].plot(test_t, test_target[0,0,:], "k--")
ax[1].plot(test_t, states[:,::ds,:].reshape(-1, test_n_steps).T)

ax[0].set_title("Output")
ax[1].set_title("Select states")
fig.tight_layout()
plt.show()