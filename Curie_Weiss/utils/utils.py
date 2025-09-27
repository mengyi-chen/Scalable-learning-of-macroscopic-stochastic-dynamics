import numpy as np 
import random
import os 
import torch
import torch.nn.functional as F
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

        z0_train = []
        z1_train = []
        z0_train_naive = []
        z1_train_naive = [] 
        idx_train_partial = []
        time_step_train = []
        for i in tqdm(range(data.shape[0])):
            X0 = data[i] # [B, L, L]
            sum_spin = torch.sum(X0, dim=(1, 2), keepdim=True)
            z0 = torch.mean(X0, dim=(1, 2)) # [B]

            X0 = X0.reshape(-1, d, box_L, d, box_L) # [B, 2, 16, 2, 16]
            X0 = X0.permute(0, 1, 3, 2, 4) # [B, 2, 2, 16, 16]
            X0 = X0.flatten(1, 2) # [B, 4, 16, 16]
            idx = torch.randint(0, d ** 2, (X0.shape[0],), device=X0.device)

            X0_partial = X0[torch.arange(X0.shape[0]), idx].clone() # [batch_size, 16, 16]
            z0_partial = torch.mean(X0_partial, dim=(1, 2)) # [B]

            X1_partial, glauber_time = glauber_continuous(X0_partial.clone(), 1.0 / self.T, self.h, n_events, \
                                                          partial=True, sum_spin=sum_spin, large_L=L)
                                                          

            z1_partial = torch.mean(X1_partial, dim=(1, 2)) # [B]
            z1 = z0 + (z1_partial - z0_partial) # [B]
            z0_train.append(z0)
            z1_train.append(z1)
            idx_train_partial.append(idx)
            time_step_train.append(glauber_time)

            z0_naive = z0
            z1_naive = z1_partial
            z0_train_naive.append(z0_naive)
            z1_train_naive.append(z1_naive)

        z0_train = torch.stack(z0_train) # [N, B]
        z1_train = torch.stack(z1_train) # [N, B]

        z0_train_naive = torch.stack(z0_train_naive) # [N, B]
        z1_train_naive = torch.stack(z1_train_naive) # [N, B]

        idx_train_partial = torch.stack(idx_train_partial) # [N, B]
        time_step_train = torch.stack(time_step_train) # [N, B]

        return z0_train, z1_train, time_step_train, z0_train_naive, z1_train_naive, idx_train_partial



def scaleup_batched_torch(spins: torch.Tensor, repeat: int) -> torch.Tensor:
    """2D version of scaleup_batched_torch"""
    B, L, _ = spins.shape
    
    # Create repeatxrepeat duplication using repeat_interleave
    # Duplicate along x and y dimensions
    up = spins.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2)
    
    return up


def glauber_continuous(
    spin: torch.Tensor,
    beta: Optional[float] = None,
    h: float = 0.0,
    n_events: Optional[int] = None,
    partial: bool = False,
    sum_spin: Optional[torch.Tensor] = None,
    large_L: Optional[int] = None
) -> torch.Tensor:
    """
    Batched continuous-time Glauber KMC on 2D patches for CW model.
    
    - spin: [B, L, L], batch of 2D patches
    - beta: inverse temperature
    - h: external field

    Returns: [B, L, L] batch of modified 2D spin tensors 
    """

    B = spin.shape[0]
    L = spin.shape[1]
    device = spin.device
    glauber_time = torch.zeros(B, device=device, dtype=torch.float32)
    batch_indices = torch.arange(B, device=device)

    if n_events is None:
        n_events = L ** 2
    
    if spin.dim() != 3:
        raise ValueError("spin must be a 3D tensor [B, L, L].")
    
    if partial == False:
        sum_spin = torch.sum(spin, dim=(1, 2), keepdim=True)
        large_L = L 
    else:
        assert sum_spin is not None, "sum_spin must be provided for partial update"
        assert large_L is not None, "large_L must be provided for partial update"

    for _ in range(n_events): 
        # Compute neighbor field for all 2D patches at once (proper boundary conditions)
        # Four neighbors in 2D: ±x, ±y directions

        R = (sum_spin - spin) / large_L ** 2
        dH = 2 * spin * R + 2 * h * spin  # Energy change
        rates = 1.0 / (1.0 + torch.exp(beta * dH))  # [B, L, L]

        rates_flat = rates.reshape(B, -1)  # [B, L*L]
        total_rates = rates_flat.sum(dim=1, keepdim=True)  # [B, 1]
        
        if torch.any(total_rates < 1e-10):
            raise ValueError("Total rate is too small, please check the input parameters.")

        # Normalize probabilities
        prob = rates_flat / total_rates  # [B, L*L]
        cum_probs = torch.cumsum(prob, dim=1)  # [B, L*L]

        # Generate random numbers for each batch element
        r = torch.rand(B, 1, device=device)
        selected_flat_idx = torch.searchsorted(cum_probs, r, right=False)
        selected_flat_idx = torch.clamp(selected_flat_idx, max=L * L - 1).squeeze(1)

        x = selected_flat_idx // L
        y = selected_flat_idx % L
        
        sum_spin = sum_spin - 2 * spin[batch_indices, x, y].view(B, 1, 1)  # Reshape to match sum_spin dimensions
        spin[batch_indices, x, y] *= -1

        # Update time using exponential waiting times
        rand_vals = torch.rand(B, device=spin.device)
        rand_vals = torch.clamp(rand_vals, min=1e-10, max=1.0)  # Avoid log(0)
        jumping_time = - torch.log(rand_vals) / total_rates.squeeze(1)

        # NOTE
        # jumping_time = 1 / total_rates.squeeze(1)


        glauber_time += jumping_time

    if torch.any(glauber_time == float('inf')):
        raise ValueError("glauber_time contains inf values, please check the input parameters.")

    return spin, glauber_time
    

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
