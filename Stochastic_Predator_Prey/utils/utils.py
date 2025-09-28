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
    """Right-hand side of the stochastic predator-prey equations."""
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

