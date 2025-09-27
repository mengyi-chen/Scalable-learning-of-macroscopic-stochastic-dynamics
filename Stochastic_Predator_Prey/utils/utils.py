import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as init

# Set the random seed for reproduction
def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rhs(u, v, a=3, b=0.4):
    # u: [B, Nx]
    # v: [B, Nx]
    f_u = u * (1 - u - v) 
    f_v = a * v * (u - b)
    f_u = f_u.unsqueeze(1)
    f_v = f_v.unsqueeze(1)
    f_uv = torch.cat([f_u, f_v], 1)

    return f_uv

def euler(f, x0, dt, noise_level=0.02):
    x0_prime = f(x0)
    x1 = x0 + dt * x0_prime + noise_level *  np.sqrt(dt) * torch.randn_like(x0)
    return x0, x1

def euler_partial(f, x0, dt, noise_level=0.02):
    x0_prime = f(x0)
    x1 = x0[..., 1:-1] + dt * x0_prime + noise_level *  np.sqrt(dt) * torch.randn_like(x0_prime)
    return x0, x1

def rk4(f, x0, dt, noise_level=0.02):
    k1 = dt * f(x0)
    k2 = dt * f(x0 + k1 / 2)
    k3 = dt * f(x0 + k2 / 2)
    k4 = dt * f(x0 + k3)

    x1 = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + noise_level *  np.sqrt(dt) * torch.randn_like(x0)
    return x0, x1

def rk4_partial(f, x0, dt, noise_level=0.02):
    k1 = dt * f(x0)
    k2 = dt * f(x0 + k1 / 2)
    k3 = dt * f(x0 + k2 / 2)
    k4 = dt * f(x0 + k3)

    x1 = x0[..., 1:-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + noise_level *  np.sqrt(dt) * torch.randn_like(k1)
    return x0, x1

class Net(nn.Module):
    def __init__(self, dt, device, a=3, b=0.4, D=0.):
        super(Net, self).__init__()
        # 2nd order differencing filter
        self.delta = torch.Tensor([[[1, -2, 1]]]).to(device)
        self.pad = nn.ReplicationPad1d(1) # Replication pad for boundary condition
        self.dt = dt
        self.a = a
        self.b = b
        self.D = D
        

    def forward(self, x0):
        # x0: [B, 2, Nx]
        f_uv = rhs(x0[:,0], x0[:,1], a=self.a, b=self.b) # f_uv: [B, 2, Nx]
        u_pad = self.pad(x0) # [B, 2, Nx + 2]
   
        diffusion_u = self.D * F.conv1d(u_pad[:,0].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 
        diffusion_v = F.conv1d(u_pad[:,1].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 
        
        diffusion = torch.cat([diffusion_u, diffusion_v], 1)
        x0_prime = diffusion + f_uv # [B, 2, Nx]

        return x0_prime
    

class NetPartial(nn.Module):
    def __init__(self, dt, device, a=3, b=0.4, D=0.):
        super(NetPartial, self).__init__()
        # 2nd order differencing filter
        self.delta = torch.Tensor([[[1, -2, 1]]]).to(device)
        self.dt = dt
        self.a = a
        self.b = b
        self.D = D
        

    def forward(self, x0):
        # x0: [B, 2, nx+2] with ghost boundary
        f_uv = rhs(x0[:, 0, 1:-1], x0[:, 1, 1:-1], a=self.a, b=self.b) # f_uv: [B, 2, nx]

        diffusion_u = self.D * F.conv1d(x0[:,0].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 
        diffusion_v = F.conv1d(x0[:,1].unsqueeze(1), self.delta) # diffusion term [B, 2, Nx] 

        diffusion = torch.cat([diffusion_u, diffusion_v], 1)

        # \frac{\partial u}{\partial t} = u(1-u-v)
        # \frac{\partial v}{\partial t} = av(u-b) + \frac{\partial^2 v}{\partial^ x}
        x0_prime = diffusion + f_uv # [B, 2, Nx]

        return x0_prime



class Autoencoder(nn.Module):
    def __init__(self, grid_size, hidden_dim, n_parts=10):
        super(Autoencoder, self).__init__()
        self.input_dim = grid_size 
        self.hidden_dim = hidden_dim
        self.n_parts = n_parts
        assert grid_size % n_parts == 0, "Grid size must be divisible by number of parts"
        self.part_size = grid_size // n_parts

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(self.part_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim // 2),
        )

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim * 2),
        )
        self.register_buffer('mean', torch.zeros(2 + hidden_dim), persistent=False)
        self.register_buffer('std', torch.ones(2 + hidden_dim), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        
    def cal_macro(self, x):
        # x: [B, 2, grid_size]
        z_macro = torch.mean(x, -1)
        return z_macro

    def encoder(self, x):
        # x shape: [B, 2, n_dim]
        z_macro = self.cal_macro(x)  # [B, 2] 

        x = x.reshape(-1, 2, self.n_parts, self.part_size) # [B, 2, n_parts, part_size]
        # x = x.permute(0, 2, 1, 3).flatten(2, 3) # [B, n_parts, 2 * part_size]
        z_val = self.encoder_net(x)  # [B, 2, n_parts, hidden_dim // 2]
        z_val = torch.mean(z_val, dim=2).flatten(1, 2) # [b, 2, hidden_dim // 2]

        z = torch.cat([z_macro, z_val], -1)  # [B, hidden_dim + 2]
        return z

    def encode(self, x0, x1, partial=False, index=None):

        if partial == False:
            z0 = self.encoder(x0)  # [B, macro_dim + hidden_dim]
            z1 = self.encoder(x1) # [B, macro_dim + hidden_dim]

            z0 = (z0 - self.mean) / (self.std) # [B, macro_dim + hidden_dim]
            z1 = (z1 - self.mean) / (self.std) # [B, macro_dim + hidden_dim]
            return z0, z1
        
        if partial == True:
            assert index is not None, "index must be provided when partial is True"
            assert index.shape[0] == x0.shape[0], "index must have the same batch size as x"

            z0 = self.encoder(x0)

            x0 = x0.reshape(-1, 2, self.n_parts, self.part_size)
            # x0 = x0.permute(0, 2, 1, 3) 
            batch_indices = torch.arange(x0.shape[0], device=x0.device)

            x0_partial = x0[batch_indices, :, index] # [B, 2, part_size]
            x1_partial = x1 # [B, 2, part_size]
        
            z0_macro_partial = self.cal_macro(x0_partial) # [B, 2]
            z1_macro_partial = self.cal_macro(x1_partial) # [B, 2] 
            z0_hat_partial = self.encoder_net(x0_partial).flatten(1, 2) # [B, hidden_dim]
            z1_hat_partial = self.encoder_net(x1_partial).flatten(1, 2) # [B, hidden_dim]

            z0_partial = torch.cat([z0_macro_partial, z0_hat_partial], -1) # [B, macro_dim + hidden_dim]
            z1_partial = torch.cat([z1_macro_partial, z1_hat_partial], -1) # [B, macro_dim + hidden_dim]

            z1_hat = z0 + (z1_partial - z0_partial) # [B, macro_dim + hidden_dim]

            z0 = (z0 - self.mean) / (self.std)
            z1_hat = (z1_hat - self.mean) / (self.std) # [B, macro_dim + hidden_dim]

            z0_naive = z0
            z1_naive = z1_partial 
            z1_naive = (z1_naive - self.mean) / (self.std)

            return z0, z1_hat, z0_naive, z1_naive
        

    def decoder(self, z):
        x = self.decoder_net(z)
        x = x.view(-1, 2, self.input_dim)
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
        
