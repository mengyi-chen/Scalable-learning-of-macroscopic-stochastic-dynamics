import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from tqdm import tqdm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, closure_dim=2, macro_dim=2, L=64, patch_L=16):
        super().__init__()
        assert L % patch_L == 0, "L must be divisible by patch_L"

        self.closure_dim = closure_dim
        self.macro_dim = macro_dim
        self.L = L
        self.patch_L = patch_L
        self.d = int(L / patch_L)
        
        num_downsamples = int(np.log2(patch_L // 2))
        self.num_downsamples = num_downsamples
        layers = []
        in_channels = 1
        out_channels = 32 
        output_size = patch_L 
        
        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.MaxPool2d(2, stride=2))

            in_channels = out_channels
            out_channels *= 2
            output_size //= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * output_size ** 2, closure_dim))
        self.model = nn.Sequential(*layers)

        self.register_buffer('min_val', torch.zeros(macro_dim + closure_dim), persistent=False)
        self.register_buffer('max_val', torch.ones(macro_dim + closure_dim), persistent=False)
        
        self.apply(weights_init)
    
    def cal_macro(self, x):
        """"
        Calculate the macroscopic variables: magnetization and domain wall density
        
        Args:
            x (Tensor): Input tensor of shape [B, 1, L, L]
        
        Returns:
            z_macro (Tensor): Macroscopic variables of shape [B, macro_dim]
        """
        # magnetization 
        mag = torch.mean(x, dim=(1, 2, 3)) # [B]

        # # domain wall density
        shift_x = torch.roll(x, shifts=1, dims=2)
        shift_y = torch.roll(x, shifts=1, dims=3)

        dw_x = 0.5 * (1 - x * shift_x)
        dw_y = 0.5 * (1 - x * shift_y)

        total_dw = dw_x + dw_y
        rho_dw = total_dw.mean(dim=(1, 2, 3)) / 2
        z_macro = torch.stack([mag, rho_dw], dim=-1) # [B, 2]
        return z_macro
            
    def forward(self, x):  
        """Encode the input state into latent representation

        Args:
            x (Tensor): Input tensor of shape [B, 1, L, L].

        Returns:
            Tensor: Latent representation of shape of shape [B, macro_dim + closure_dim].
        """
        # Calculate macroscopic variables
        z_macro = self.cal_macro(x) # [B, 1, L, L] -> [B, macro_dim]

        # Calculate closure variables
        # \hat{z} = \frac{1}{K} \sum_{I} \hat{z}_{I}
        # partition x into patches
        x = x.reshape(-1, self.d, self.patch_L, self.d, self.patch_L)
        x = x.permute(0, 1, 3, 2, 4) # [B, 2, 2, 8, 8]
        x = x.flatten(0, 2).unsqueeze(1) # [B * 4, 1, 8, 8]
        z_hat = self.model(x).reshape(-1, self.d ** 2, self.closure_dim) # [B, 4, closure_dim]
        z_hat = torch.mean(z_hat, dim=1) # [B, closure_dim]
        
        # Combine macroscopic and closure variables
        # z = (z^{\ast}, \hat{z})
        z = torch.cat([z_macro, z_hat], -1) # [B, macro_dim + closure_dim]
        return z
    
    def encode_pairs(self, x0, x1, partial=False, index=None):
        """Encode pairs of input tensors.

        Args:
            x0 (Tensor): Input tensor of shape [B, 2, grid_size]
            x1 (Tensor): Input tensor of shape [B, 2, grid_size] if partial is False, else [B, 2, patch_L]
            partial (bool, optional): Whether to use partial encoding. Defaults to False.
            index (Tensor, optional): Indices of the patches to use for partial encoding. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Encoded representations of the input tensors.
        """

        if partial == False: 
            # Full evolution case
            z0 = self.forward(x0) # [B, macro_dim + closure_dim]
            z1 = self.forward(x1) # [B, macro_dim + closure_dim]

            z0 = (z0 - self.min_val) / (self.max_val - self.min_val) # [B, macro_dim + closure_dim]
            z1 = (z1 - self.min_val) / (self.max_val - self.min_val) # [B, macro_dim + closure_dim]
            return z0, z1

        if partial == True:
            # Partial evolution case
            assert index is not None, "index must be provided when partial is True"
            assert index.shape[0] == x0.shape[0], "index must have the same batch size as x"

            # ========== our method ==========
            # z0 = \varphi(x0)
            z0 = self.forward(x0)

            # $x_{0, I}$
            x0 = x0.reshape(-1, self.d, self.patch_L, self.d, self.patch_L)
            x0 = x0.permute(0, 1, 3, 2, 4) # [B, 2, 2, 32, 32]
            x0 = x0.flatten(1, 2) # [B, 4, 32, 32]
            batch_indices = torch.arange(x0.shape[0], device=x0.device)
            x0_partial = x0[batch_indices, index].unsqueeze(1) # [B, 1, 32, 32] 
            
            # $x_{1, I}$
            x1_partial = x1 

            z0_macro_partial = self.cal_macro(x0_partial) # [B-1, 1, 16, 16, 16] -> [B-1, macro_dim]
            z1_macro_partial = self.cal_macro(x1_partial) # [B-1, 1, 16, 16, 16] -> [B-1, macro_dim]
            z0_hat_partial = self.model(x0_partial) # [B-1, 1, 16, 16, 16] -> [B-1, closure_dim]
            z1_hat_partial = self.model(x1_partial) # [B-1, 1, 16, 16, 16] -> [B-1, closure_dim]

            # $\varphi(x_{0, I})$
            z0_partial = torch.cat([z0_macro_partial, z0_hat_partial], -1) # [B-1, macro_dim + closure_dim]
            # $\varphi(x_{1, I})$
            z1_partial = torch.cat([z1_macro_partial, z1_hat_partial], -1) # [B-1, macro_dim + closure_dim]
            
            # z1 = \varphi(x_0) + (\varphi(x_{1, I}) - \varphi(x_{0, I}))
            z1_hat = z0 + (z1_partial - z0_partial) # [B-1, macro_dim + closure_dim]

            z0 = (z0 - self.min_val) / (self.max_val - self.min_val)
            z1_hat = (z1_hat - self.min_val) / (self.max_val - self.min_val) # [B-1, macro_dim + closure_dim]

            return z0, z1_hat


class Decoder(nn.Module):
    def __init__(self, closure_dim=2, macro_dim=2, L=64, patch_L=16): # Added L
        super().__init__()
        assert L % patch_L == 0, "L must be divisible by patch_L"
        
        self.macro_dim = macro_dim
        self.closure_dim = closure_dim
        self.L = L
        self.patch_L = patch_L
        self.d = int(L / patch_L)

        # Calculate number of upsampling layers needed to reach L from initial 4x4
        if L == 48:
            num_upsamples = int(np.log2(L // 3))
            initial_size = (128, 3, 3)
        else:
            num_upsamples = int(np.log2(L // 4))
            initial_size = (128, 4, 4)

        layers = []
        
        # Initial linear layer and unflatten
        layers.append(nn.Linear(self.closure_dim + self.macro_dim, np.prod(initial_size)))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Unflatten(1, initial_size))  # [B, 128, 4, 4]
        
        # Build upsampling layers dynamically
        in_channels = 128
        current_size = 4
        
        for i in range(num_upsamples):
            out_channels = max(16, in_channels // 2) if i < num_upsamples - 1 else 1
            
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_size *= 2
            
            # Add batch norm and activation for all layers except the last one
            if i < num_upsamples - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.2))
            else:
                # Final layer uses sigmoid activation
                layers.append(nn.Sigmoid())
            
            in_channels = out_channels
        
        self.model = nn.Sequential(*layers)

        # Ensure min/max are registered correctly
        self.apply(weights_init)

    def forward(self, z):
        """Decode the latent representation back to the original state space

        Args:
            z (Tensor): Input tensor of shape [B, macro_dim + closure_dim]

        Returns:
            Tensor: Output tensor of shape [B, 1, L, L]
        """

        x = self.model(z) 

        # map the output from [0, 1] to [-1, 1] since the Ising state is represented by -1 and 1
        x = x * 2 - 1
        return x

    
class Conv2DAutoencoder(nn.Module):
    def __init__(self, closure_dim=2, macro_dim=2, L=16, patch_L=8):
        super().__init__()
        self.closure_dim = closure_dim
        self.macro_dim = macro_dim
        
        self.encoder = Encoder(closure_dim, macro_dim, L, patch_L)
        self.decoder = Decoder(closure_dim, macro_dim, L, patch_L)
    
    def forward(self, x):
        """Autoencoder for identifying closure variables
        
        Args:
            x (Tensor): Input tensor of shape [B, 1, L, L]

        Returns:
            Tensor: Reconstructed tensor of shape [B, 1, L, L]
        """
        # x: [B, 1, L, L]
        z = self.encoder(x)
        x = self.decoder(z) 
        return x
    
    def encode_pairs(self, x0, x1, partial=False, index=None):
        z = self.encoder.encode_pairs(x0, x1, partial=partial, index=index)
        return z
        
    def cal_macro(self, x):
        z_macro = self.encoder.cal_macro(x)
        return z_macro



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
        