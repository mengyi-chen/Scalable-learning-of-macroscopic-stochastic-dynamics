import numpy as np 
import random
import torch
import os 
import torch.nn as nn
import torch.optim as optim

def set_seed(seed):
    """Set the seed for reproducibility across NumPy, Python, and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_mag_susceptibility(M, L, T=2.27):
    M = np.abs(M)  
    # mag_susceptibility = np.mean(M ** 2, axis=-1) - np.mean(M, axis=-1) ** 2
    mag_susceptibility = np.mean(M ** 2) - np.mean(M) ** 2
    mag_susceptibility = mag_susceptibility * L ** 2 / T 
    return mag_susceptibility

def glauber_continuous(spin, L, beta, h=0):
    """continuous time Glauber dynamics for Ising model"""
    # spin: [B, L, L]

    B = spin.shape[0]
    glauber_time = torch.zeros(B, device=spin.device, dtype=torch.float32)

    for _ in range(L * L):
        # Neighbor sum using periodic boundary conditions
        R = torch.roll(spin, 1, dims=1) + torch.roll(spin, -1, dims=1) + \
            torch.roll(spin, 1, dims=2) + torch.roll(spin, -1, dims=2)

        dH = 2 * spin * R + 2 * h * spin  # Energy change
        rates = 1.0 / (1.0 + torch.exp(beta * dH))  # [B, L, L]

        rates_flat = rates.view(B, -1)  # [B, L*L]
        total_rate = rates_flat.sum(dim=1, keepdim=True)  # [B, 1]

        # Safety check for very small total rates
        if torch.any(total_rate < 1e-10):
            raise ValueError("Total rate is too small, please check the input parameters.")

        prob = rates_flat / total_rate  # [B, L*L]
        cum_probs = torch.cumsum(prob, dim=1)  # [B, L*L]

        r = torch.rand(B, 1, device=spin.device)
        selected_flat_idx = torch.searchsorted(cum_probs, r, right=False)
        selected_flat_idx = torch.clamp(selected_flat_idx, max=L * L - 1).squeeze(1)

        # Convert flat index to 2D indices
        x = selected_flat_idx // L
        y = selected_flat_idx % L

        # Flip the selected spin for each batch
        spin[torch.arange(B), x, y] *= -1

        # Update time using exponential waiting times
        # rand_vals = torch.rand(B, device=spin.device)
        # rand_vals = torch.clamp(rand_vals, min=1e-10, max=1.0)  # Avoid log(0)
        # jumping_time = - torch.log(rand_vals) / total_rate.squeeze(1)

        # NOTE:
        jumping_time = 1 / total_rate.squeeze(1)  # Use the total rate directly for simplicity
        glauber_time += jumping_time

    if torch.any(glauber_time == float('inf')):
        raise ValueError("glauber_time contains inf values, please check the input parameters.")

    return spin, glauber_time
