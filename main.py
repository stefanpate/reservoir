from esn import esn
import numpy as np
import torch
import time
import matplotlib.pyplot as plt


n_inputs = 1
n_hidden = 100
n_outputs = 1
spectral_radius = 0.8
pcon = 0.5
leak_rate = 0.4
gpu = 0 # -1 => cpu, 0 and above => gpu number
n_steps = 100 # Timesteps of simulation
pass_target = True
input_fcns = [np.sin, np.cos, lambda x: np.sin(x) + np.cos(x)]
n_samples = len(input_fcns)

# Chech I can instantiate
res = esn(n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu)
print(res)

# Check SR is same as above
evals, _ = torch.eig(res.w)
print("Spectral radius: ", (evals**2).sum(dim=1).sqrt_().max())

# Check w pcon is correct
n_zero_weights = (res.w == 0).to(dtype=torch.float).sum()
print("Cxn density: ", (n_hidden**2 - n_zero_weights) / n_hidden**2)

# Create n_samples samples of input
t = np.linspace(0, 2 * np.pi, num=n_steps)
input = []
for f in input_fcns:
    input.append(f(t).reshape(n_inputs, n_steps))
input = np.stack(input, axis=0) # (3, n_inputs, n_steps)

# Create n_samples of target
target = []
for f in input_fcns[-1::-1]:
    target.append(f(t).reshape(n_outputs, n_steps))
target = np.stack(target, axis=0)

# Simulate. Measure compute time
start = time.time()

if pass_target:
    states, outputs = res.simulate(input, target=target)
else:
    states, outputs = res.simulate(input)

end = time.time()
elapsed = end - start

if gpu == -1:
    device = 'CPU'
else:
    device = torch.cuda.get_device_name(gpu)

print(f"Device: {device}, Time elapsed = {elapsed:.4f} seconds")


# Plot simulation
ds = 10 # Downsample
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
ax[0].plot(t, input[:,0,:].T)
ax[1].plot(t, outputs[:,0,:].T)
ax[2].plot(t, states[:,::ds,:].reshape(-1, n_steps).T)

ax[0].set_title("Input")
ax[1].set_title("Output")
ax[2].set_title("Select states")
fig.tight_layout()
plt.show()

# Tune w_out
w_out_hat_torch = res.fit(input, target)
w_out_hat = w_out_hat_torch.cpu().numpy()
w_out = res.w_out.cpu().numpy()
plt.hist(w_out.reshape(-1))
plt.hist(w_out_hat.reshape(-1), alpha=0.5)
plt.show()

res.set_w_out(w_out_hat_torch) # Set w_out

states, outputs = res.simulate(input) # Simulate again

# Plot after tuning w_out
ds = 10 # Downsample
states, outputs = states.cpu().numpy(), outputs.cpu().numpy()

fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
ax[0].plot(t, input[:,0,:].T)
ax[1].plot(t, outputs[:,0,:].T)
ax[2].plot(t, states[:,::ds,:].reshape(-1, n_steps).T)

ax[0].set_title("Input")
ax[1].set_title("Output")
ax[2].set_title("Select states")
fig.tight_layout()
plt.show()