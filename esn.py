import numpy as np
import torch

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
        
        # Adjacency matrix
        self.w = torch.rand(size=(self.n_hidden, self.n_hidden), device=self.device) * 2 - 1
        zero_out_mask = torch.rand(size=(self.n_hidden, self.n_hidden), device=self.device) < self.pcon
        zero_out_mask = zero_out_mask.to(dtype=torch.float)
        self.w.mul_(zero_out_mask)
        w_evals, _ = torch.eig(self.w)
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


    def simulate(self, input, target=None):
        '''
        Simulate n_steps of ESN dynamics.

        Args:
            - input: Tensor / numpy array of inputs (n_samples, n_inputs, n_timesteps)
            - target: (Optional) Tensor / numpy array of target signal (n_samples, n_outputs, n_timesteps)

        Returns:
            - Tensors of states & outputs
        '''
        scl_x_init = 10 # Divides normal rand var that inits x
        
        # Convert non-torch-tensor input
        if not torch.is_tensor(input):
            input = torch.tensor(input, dtype=torch.float, device=self.device)

        # Infer # timesteps & # samples from input
        n_steps = input.size(dim=-1)
        n_samples = input.size(dim=0)
        
        # Convert non-torch-tensor target
        if (target is not None) & (not torch.is_tensor(target)):
            target = torch.tensor(target, dtype=torch.float, device=self.device)

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

            # Integrate one step. If provided target, pass as feedback,
            # else provide output as feedback
            if target is None:
                x, y = self._one_step(x_prev, input[:,:,i], y_prev, n_samples)
            else:
                x, y = self._one_step(x_prev, input[:,:,i], target[:,:,i], n_samples)
            
            x_prev, y_prev = x, y # Update variables

        return states, outputs


    def fit(self, input, target):
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float, device=self.device)

        states, _ = self.simulate(input, target=target) # Simulate

        # Flatten 3D tensors into matrices
        target = target.transpose(0, 1).reshape(self.n_outputs, -1)
        states = states.transpose(0, 1).reshape(self.n_hidden, -1)

        st = states.size(dim=-1) # Infer n_samples * n_steps
        states = torch.cat((torch.ones(size=(1, st), device=self.device), states), dim=0) # Concat 1s col
        states = states.cpu().numpy() # Convert to numpy array

        # Compute pseudoinverse
        states_pinv = np.linalg.pinv(states)
        states_pinv = torch.tensor(states_pinv, dtype=torch.float, device=self.device)

        w_out_hat = torch.matmul(target, states_pinv)

        return w_out_hat

    def set_w_out(self, w_out):
        if not torch.is_tensor(w_out):
            w_out = torch.tensor(w_out, dtype=torch.float, device=self.device)

        if w_out.shape != self.w_out.shape:
            print("Shape mismatch w_out")
        else:
            self.w_out = w_out


