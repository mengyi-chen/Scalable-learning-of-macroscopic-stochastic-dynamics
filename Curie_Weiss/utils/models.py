import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from tqdm import tqdm


class DiffusitivityNet(nn.Module):
    """Generate diffusitivity term for SDE
    """
    def __init__(self, n_dim, mode='arbitrary'):
        super(DiffusitivityNet, self).__init__()
        self.n_dim = n_dim
        self.mode = mode

        if mode == 'arbitrary':
            self.output_layer = nn.Sequential(
                nn.Linear(n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim * n_dim),
            )

        elif mode in ['diagonal']:
            self.output_layer = nn.Sequential(
                nn.Linear(n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim)
            )

        elif mode == 'constant_diagonal':
            self.kernel = nn.Parameter(torch.ones(n_dim))  # Parameters are automatically considered for optimization
        
        elif mode == 'constant':
            self.kernel = nn.Parameter(torch.ones(n_dim * n_dim))

        else:
            raise ValueError(f"Mode {mode} is unknown")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """Diffusion term 
        Args:
            x (Tensor): Input tensor of shape [B, n_dim]

        Returns:
            Tensor: Output tensor of shape [B, n_dim, n_dim]
        """        
        # General case: arbitrary diffusion matrix
        if self.mode == 'arbitrary':

            output = self.output_layer(x)
            output = output.view(-1, self.n_dim, self.n_dim)
        
        # Diagonal diffusion matrix
        elif self.mode == 'diagonal':
            
            output = self.output_layer(x)
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        # Constant diagonal diffusion matrix
        elif self.mode == 'constant_diagonal':
            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        # Constant diffusion matrix
        elif self.mode == 'constant':
            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = output.view(-1, self.n_dim, self.n_dim) 

        return output

class SDE_Net(nn.Module):
    def __init__(self, delta_t, mode='constant_diagonal', n_dim=3, epsilon=1e-4):
        super().__init__()
        self.delta_t = nn.Parameter(torch.tensor(delta_t), requires_grad=False)
        self.n_dim = n_dim
        self.mode = mode
        # Drift network
        self.drift_net = nn.Sequential(
            nn.Linear(n_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_dim)
        )
        self.epsilon = epsilon
        
        # Diffusitivity network
        self.sigma_net = DiffusitivityNet(n_dim=n_dim, mode=mode)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.uniform_(module.bias, 0, 0.1)
    
    def drift(self, x):
        # x: [B, 3]
        drift = self.drift_net(x)

        return drift
    
    def custom_loss(self, x0, x1, dt=None, coeff=1):
        """Compute the custom loss between two states.
            Loss = (x1 - x0 - drift * dt)^T * (K \Sigma)^{-1} * (x1 - x0 - drift * dt) + log|K \Sigma|
        
        Args:
            x0 (Tensor): The initial state. Shape: [B, n_dim]
            x1 (Tensor): The target state. Shape: [B, n_dim]
            dt (float, optional): The time step size. If None, uses self.delta_t. Defaults to None.
            coeff (int, optional): hyperparameter to scale the covariance matrix. Defaults to 1.

        Returns:
            Tensor: The computed loss.
        """
        if dt == None:
            dt = self.delta_t

        drift = self.drift(x0) # [B, 3]
        sigma = self.sigma_net(x0) # [B, 3, 3]

        cov_matrix = torch.bmm(sigma, sigma.transpose(1, 2)) * dt.view(-1, 1, 1)
        # add small value to the diagonal for numerical stability
        Sigma = cov_matrix * coeff + torch.eye(self.n_dim, device=x0.device) * self.epsilon
        Sigma_inv = torch.linalg.inv(Sigma)

        X = x1 - x0 - drift * dt
        a1 = torch.einsum('ij,ijk,ik->i', X, Sigma_inv, X)
        a1 = a1.view(-1, 1)

        a2 = torch.linalg.slogdet(Sigma)[1]  # Returns a tuple (sign, logdet); we need logdet
        a2 = a2.view(-1, 1)

        return torch.mean(a1 + a2)
    
    def predict(self, x, steps, dt=None):
        """Predict the long-term trajectory for a given number of steps.
        
        Args:
            x (Tensor): The initial state. Shape: [B, n_dim]
            steps (int): The number of steps to predict.
            dt (float, optional): The time step size. If None, uses self.delta_t. Defaults to None.
        
        Returns:
            Tensor: The predicted trajectory. Shape: [B, n_dim, steps]
        """
        if dt == None:
            dt = self.delta_t
        
        # x: [B, latent_dim]
        predict_tra = [x]
        for _ in tqdm(range(steps-1)):

            x0 = predict_tra[-1]
            drift = self.drift(x0) # [B, 3]
            sigma = self.sigma_net(x0) # [B, 3, 3]
            delta_W = torch.normal(mean=0., std=torch.sqrt(dt), size=(x.shape[0],self.n_dim), device=x.device)
            x1 = x0 + drift * dt + torch.einsum('ijk,ik->ij', sigma, delta_W)

            predict_tra.append(x1)

        predict_tra = torch.stack(predict_tra, 2).transpose(2, 1)
        return predict_tra
        
    def forward(self, z0, noise):
        """Euler-Maruyama scheme for one step

        Args:
            z0 (Tensor): The initial state.
            noise (Tensor): The noise to be added.

        Returns:
            Tensor: The next state.
        """
        # z0: [B, D]
        # noise: [B, D]
        drift = self.drift(z0) # [B, D]
        sigma = self.sigma_net(z0) # [B, D, D]

        z1 = z0 + drift * self.delta_t + torch.einsum('ijk,ik->ij', sigma, noise) * torch.sqrt(self.delta_t)
        return z1
        