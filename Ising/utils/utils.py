import numpy as np 
import random
import torch
import os 
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from tqdm import tqdm

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

def glauber_continuous(spin, beta, h=0, n_events=None):
    """continuous time Glauber dynamics for Ising model"""
    # spin: [B, L, L]

    B = spin.shape[0]
    L = spin.shape[1]
    glauber_time = torch.zeros(B, device=spin.device, dtype=torch.float32)
    if n_events is None:
        n_events = L * L
        
    # for _ in range(L * L):
    for _ in range(n_events):
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
        rand_vals = torch.rand(B, device=spin.device)
        rand_vals = torch.clamp(rand_vals, min=1e-10, max=1.0)  # Avoid log(0)
        jumping_time = - torch.log(rand_vals) / total_rate.squeeze(1)
        
        # NOTE:
        # jumping_time = 1 / total_rate.squeeze(1)  # Use the total rate directly for simplicity
        glauber_time += jumping_time

    if torch.any(glauber_time == float('inf')):
        raise ValueError("glauber_time contains inf values, please check the input parameters.")

    return spin, glauber_time

def scaleup_batched_torch(spins: torch.Tensor, repeat: int) -> torch.Tensor:
    """2D version of scaleup_batched_torch"""
    B, L, _ = spins.shape
    
    # Create repeatxrepeat duplication using repeat_interleave
    # Duplicate along x and y dimensions
    up = spins.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2)
    
    return up


def patchwise_relax_batched_torch(
    spins: torch.Tensor,
    T: Optional[float] = None,
    patch_size: int = 16,
    stride: int = 8,
    h: float = 0.0,
    sweep: int = 1,
    n_events: Optional[int] = None
) -> torch.Tensor:
    """
    Batched version of patchwise relaxation using CT-Glauber KMC for 2D systems.
    
    - spins: [B, L, L], batch of 2D spin configurations
    - T: temperature
    - patch_size: size of square patches to process

    - h: external field
    
    Returns: [B, L, L] batch of relaxed 2D spin configurations
    """
    device = spins.device
    if spins.dim() != 3:
        raise ValueError("spins must be a 3D tensor [B, L, L].")
    if not (spins.shape[1] == spins.shape[2]):
        raise ValueError("spins must have square spatial dimensions [B, L, L].")
    if not torch.all(torch.isin(spins, torch.tensor([-1, 1], device=device))):
        raise ValueError("spins entries must be ±1.")
    
    B, L, _ = spins.shape
    if L < patch_size or L % stride != 0:
        raise ValueError("L must allow sliding patches with the given patch_size and stride.")

    starts = range(0, L, stride)
    beta = 1.0 / T
    if n_events is None:
        n_events = patch_size * patch_size

    for _ in range(sweep):
        for i in starts:
            for j in starts:
                x_indices = torch.arange(i, i + patch_size, device=device) % L
                y_indices = torch.arange(j, j + patch_size, device=device) % L

                # Extract 2D patches for all batches using advanced indexing
                patch_spins = spins[:, x_indices[:, None], y_indices[None, :]]
                
                # Process all patches in the batch at once
                updated_patches, _ = glauber_continuous(
                    patch_spins, beta=beta, h=h, n_events=n_events
                )

                spins[:, x_indices[:, None], y_indices[None, :]] = updated_patches
    return spins

class PartialGenerator():

    def __init__(self, T, h):
        self.T = T
        self.h = h

    def generate(self, data, box_L, n_events):
        # data shape: [N, B, L, L] for 2D
        
        L = data.shape[2]
        data = data.to(torch.float32)
        assert L % box_L == 0, "L must be a multiple of box_L"
        d = int(L // box_L)

        X1_train_partial = [] 
        idx_train_partial = []
        time_step_train = []
        for i in tqdm(range(data.shape[0])):
            X0 = data[i] # [B, L, L]

            X0 = X0.reshape(-1, d, box_L, d, box_L) # [B, 2, 16, 2, 16]
            X0 = X0.permute(0, 1, 3, 2, 4) # [B, 2, 2, 16, 16]
            X0 = X0.flatten(1, 2) # [B, 4, 16, 16]
            idx = torch.randint(0, d ** 2, (X0.shape[0],), device=X0.device)

            X0_partial = X0[torch.arange(X0.shape[0]), idx].clone() # [batch_size, 16, 16]
            X1_partial, glauber_time = glauber_continuous(X0_partial.clone(), 1.0 / self.T, self.h, n_events)

            X1_train_partial.append(X1_partial)
            idx_train_partial.append(idx)
            time_step_train.append(glauber_time)

        X1_train_partial = torch.stack(X1_train_partial) # [N, B, L, L]
        idx_train_partial = torch.stack(idx_train_partial) # [N, B]
        time_step_train = torch.stack(time_step_train) # [N, B]

        return X1_train_partial, time_step_train, idx_train_partial
