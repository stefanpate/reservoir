import numpy as np
import torch
from sklearn.linear_model import Ridge, Lasso

class esn:

    def __init__(self, n_inputs, n_hidden, n_outputs, spectral_radius, pcon, leak_rate, gpu):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.pcon = pcon
        self.leak_rate = leak_rate

        # Set hardware device in torch
        if gpu < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f"cuda:{gpu}")

        self.init_weights()


    def init_weights(self):
        '''
        Initialize weights.
        '''
        # Adjacency matrix
        self.w = torch.rand(size=(self.n_hidden, self.n_hidden), device=self.device) * 2 - 1
        zero_out_mask = torch.rand(size=(self.n_hidden, self.n_hidden), device=self.device) < self.pcon
        zero_out_mask = zero_out_mask.to(dtype=torch.float)
        self.w.mul_(zero_out_mask)
        w_evals, _ = torch.eig(self.w)
        # Eig returns nx2 array of real numbers representing real component in first column
        # and imaginary component in second column. To get abs value, square elementwise, 
        # sum along axis 1, and take square root. Note this is different than how numpy does complex numbers
        init_spectral_radius = (w_evals**2).sum(dim=1).sqrt_().max()
        self.w.mul_(self.spectral_radius / init_spectral_radius)

        self.w_in = torch.rand(size=(self.n_hidden, self.n_inputs + 1), device=self.device) * 2 - 1 # Input
        self.w_out = torch.rand(size=(self.n_outputs, self.n_hidden + 1), device=self.device) * 2 - 1 # Output
        self.w_fb = torch.rand(size=(self.n_hidden, self.n_outputs), device=self.device) * 2 - 1 # Output feedback


    def _one_step(self, x_prev, u, fb_prev, n_samples):
        '''
        Update system by one discrete time step.

        Args:
            - x_prev: State at t - 1
            - u: Input at t
            - fb_prev: Feedback term at t - 1, either target or system output
            - n_samples: Number time series in the batch
        
        Returns:
            - x: Next state
            - y: Next output
        '''
        dx = torch.tanh( u.matmul(self.w_in.t()) + x_prev.matmul(self.w.t()) + fb_prev.matmul(self.w_fb.t()) )
        x = (1 - self.leak_rate) * x_prev + self.leak_rate * dx
        x_aug = torch.cat((torch.ones(size=(n_samples, 1), device=self.device), x), dim=1)
        y = torch.matmul(x_aug, self.w_out.t())
        return x, y


    def simulate(self, n_steps, n_samples, input=None, target=None, extend=None):
        '''
        Simulate n_steps of ESN dynamics. Can optionally give input, target. If given, 
        target is provided as feedback. If extend provided, output feedback replaces 
        target feedback after end of target trace.

        Args:
            - n_steps: Number of timesteps to simulate
            - n_samples: Number of time series samples 
            - input: (Optional) Tensor / numpy array of inputs (n_samples, n_inputs, n_timesteps)
            - target: (Optional) Tensor / numpy array of target signal (n_samples, n_outputs, n_timesteps)
            - extend: (Optional) Number of timesteps to extend beyond teacher forcing

        Returns:
            - States [n_samples, n_hidden, n_steps]
            - Outputs [n_samples, n_ouputs, n_steps]
        '''
        scl_x_init = 10 # Divides normal rand var that inits x
        
        # Convert non-torch-tensor input
        if (input is not None) & (not torch.is_tensor(input)):
            input = torch.tensor(input, dtype=torch.float, device=self.device)

            # Check input dimensions
            if (n_steps != input.size(dim=-1)) | (n_samples != input.size(dim=0)):
                raise Exception("n_steps or n_samples does not match input dimensions")
        
        # Convert non-torch-tensor target
        if (target is not None) & (not torch.is_tensor(target)):
            target = torch.tensor(target, dtype=torch.float, device=self.device)

            # Check target dimensions. In extend mode, target need not go for full trial
            if (extend is None) & ((n_steps != target.size(dim=-1)) | (n_samples != target.size(dim=0))):
                raise Exception("n_steps or n_samples does not match target dimensions")

        # If no input passed, make zero tensor for _one_step
        if input is None:
            input = torch.zeros(size=(n_samples, self.n_inputs, n_steps), device=self.device)

        input = torch.cat((torch.ones(size=(n_samples, 1, n_steps), device=self.device), input), dim=1) # Concat input bias

        # Initialize state and output
        x_prev = torch.tanh(torch.randn(size=(n_samples, self.n_hidden), device=self.device) / scl_x_init)
        x_prev_aug = torch.cat((torch.ones(size=(n_samples, 1), device=self.device), x_prev), dim=1) # Concat 1s col to x
        y_prev = torch.matmul(x_prev_aug,  self.w_out.t())

        # Init tensors to store all states and outputs over time
        states = torch.empty(size=(n_samples, self.n_hidden, n_steps), device=self.device)
        outputs = torch.empty(size=(n_samples, self.n_outputs, n_steps), device=self.device)

        for i in range(n_steps):
            # Store states, outputs
            states[:,:,i] = x_prev
            outputs[:,:,i] = y_prev

            # Integrate one step
            if i < (n_steps - 1): # Avoid indexing input beyond bounds
                if target is None: # Drive with output only
                    x, y = self._one_step(x_prev, input[:,:,i+1], y_prev, n_samples)
                elif extend is not None: # Teacher force, later drive with output
                    if i < target.shape[-1]:
                        x, y = self._one_step(x_prev, input[:,:,i+1], target[:,:,i], n_samples)
                    else:
                        x, y = self._one_step(x_prev, input[:,:,i+1], y_prev, n_samples)
                else: # Teacher force
                    x, y = self._one_step(x_prev, input[:,:,i+1], target[:,:,i], n_samples)
                
                x_prev, y_prev = x, y # Update variables

        return states, outputs


    def fit(self, target, method='pinv', lam=0, input=None):
        '''
        Does linear regression to tune output weights to 
        make output close to provided target. Calls
        simulate to get outputs.

        Args:
            - target: Tensor / numpy array of target signal (n_samples, n_outputs, n_timesteps)
            - input: (Optional) Tensor / numpy array of inputs (n_samples, n_inputs, n_timesteps)
            - method: Type of regression. Defaults to pseudoinverse
            - lam: Regularization parameter for ridge and lasso regression

        Returns:
            - w_out_hat: Best output weight matrix from linear regression
        '''
        
        
        # Check whether target is tensor
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float, device=self.device)

        # Infer timesteps and samples
        n_steps = target.size(dim=-1)
        n_samples = target.size(dim=0)

        states, _ = self.simulate(n_steps, n_samples, input=input, target=target) # Simulate

        # Throw out first 1000 timesteps
        states = states[:,:,1000:]
        target = target[:,:,1000:]

        # Flatten 3D tensors into matrices
        target = target.transpose(0, 1).reshape(self.n_outputs, -1)
        states = states.transpose(0, 1).reshape(self.n_hidden, -1)

        st = states.size(dim=-1) # Infer n_samples * n_steps
        states_aug = torch.cat((torch.ones(size=(1, st), device=self.device), states), dim=0) # Concat 1s col
        
        # Convert to numpy array
        states = states.cpu().numpy()
        states_aug = states_aug.cpu().numpy()
        target = target.cpu().numpy()

        # Do regression
        if method == 'pinv':
            states_pinv = np.linalg.pinv(states_aug)
            w_out_hat = np.matmul(target, states_pinv)
        elif method == 'ridge':
            reg = Ridge(alpha=lam)
            reg.fit(states.T, target.T) # Time x features
            intercept = reg.intercept_.reshape(self.n_outputs, 1)
            coeffs = reg.coef_.reshape(self.n_outputs, self.n_hidden)
            w_out_hat = np.concatenate((intercept, coeffs), axis=1)
        elif method == 'lasso':
            reg = Lasso(alpha=lam)
            reg.fit(states.T, target.T) # Time x features
            intercept = reg.intercept_.reshape(self.n_outputs, 1)
            coeffs = reg.coef_.reshape(self.n_outputs, self.n_hidden)
            w_out_hat = np.concatenate((intercept, coeffs), axis=1)

        return w_out_hat

    def set_w_out(self, w_out):
        '''
        Set output matrix of reservoir net.
        '''
        if not torch.is_tensor(w_out):
            w_out = torch.tensor(w_out, dtype=torch.float, device=self.device)

        if w_out.shape != self.w_out.shape:
            print("Shape mismatch w_out")
        else:
            self.w_out = w_out

    def fit_rand_units(self, target, frac, input=None):
        '''
        Selects random sample of units and uses only a fraction
        for linear regression.
        '''

        # Check whether target is tensor
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float, device=self.device)

        # Infer timesteps and samples
        n_steps = target.size(dim=-1)
        n_samples = target.size(dim=0)

        states, _ = self.simulate(n_steps, n_samples, input=input, target=target) # Simulate

        # Throw out first 1000 timesteps
        states = states[:,:,1000:]
        target = target[:,:,1000:]

        # Flatten 3D tensors into matrices
        target = target.transpose(0, 1).reshape(self.n_outputs, -1)
        states = states.transpose(0, 1).reshape(self.n_hidden, -1)

        # Sample random units
        idx = np.random.rand(self.n_hidden) < frac
        # Ensure at least one unit chosen
        if not np.any(idx):
            idx = [False] * self.n_hidden
            idx[0] = True
            idx = np.array(idx)
        states = states[idx, :]

        st = states.size(dim=-1) # Infer n_samples * n_steps
        states_aug = torch.cat((torch.ones(size=(1, st), device=self.device), states), dim=0) # Concat 1s col
        
        # Convert to numpy array
        states_aug = states_aug.cpu().numpy()
        target = target.cpu().numpy()
        
        # Regression
        states_pinv = np.linalg.pinv(states_aug)
        tuned_weights = np.matmul(target, states_pinv)

        # Put tuned weights back in place, 0 otherwise
        idx = np.where(idx)[0]
        w_out_hat = np.zeros(shape=(self.n_outputs, self.n_hidden + 1))
        w_out_hat[:,0] = tuned_weights[:,0] # First tuned weight vector are biases
        for i, elt in enumerate(idx):
            w_out_hat[:, elt + 1] = tuned_weights[:, i + 1]

        return w_out_hat




