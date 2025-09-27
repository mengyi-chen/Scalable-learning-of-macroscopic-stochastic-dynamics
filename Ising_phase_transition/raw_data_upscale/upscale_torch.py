from numpy import inner
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def scaleup_batched_torch(spins: torch.Tensor, repeat: int) -> torch.Tensor:

    B, L, _ = spins.shape
    
    # Create 2x2 duplication using repeat_interleave
    # First duplicate along the height dimension, then along the width dimension
    up = spins.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2)
    
    return up


def glauber_batched_torch(
    patch_spins: torch.Tensor,
    beta: Optional[float] = None,
    h: float = 0.0,
    device: str = None, 
    n_events: Optional[int] = None
) -> torch.Tensor:
    """
    Batched continuous-time Glauber KMC on patches with proper boundary conditions.
    
    - patch_spins: [B, L+2, L+2], batch of patches with ghost spins
    - beta: inverse temperature
    - h: external field
    - device: target device for computation

    Returns: [B, L, L] batch of modified spin tensors without ghost spins
    """
    if device is None:
        device = patch_spins.device
        
    patch_spins = patch_spins.to(device)
    
    if patch_spins.dim() != 3:
        raise ValueError("patch_spins must be a 3D tensor [B, L+2, L+2].")
    
    B = patch_spins.shape[0]
    L = patch_spins.shape[1] - 2
        
    # for _ in range(L * L):
    for _ in range(n_events): 
        # Compute neighbor field for all patches at once (proper boundary conditions)
        R = patch_spins[:, :-2, 1:-1] + patch_spins[:, 2:, 1:-1] + \
            patch_spins[:, 1:-1, :-2] + patch_spins[:, 1:-1, 2:]  # [B, L, L]

        dH = 2 * patch_spins[:, 1:-1, 1:-1] * R + 2 * h * patch_spins[:, 1:-1, 1:-1]  # [B, L, L]
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
        patch_spins[torch.arange(B, device=device), x + 1, y + 1] *= -1
    
    return patch_spins[:, 1:-1, 1:-1]  # Return without ghost spins [B, L, L]

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
    Batched version of patchwise relaxation using CT-Glauber KMC.
    
    - spins: [B, L, L], batch of spin configurations
    - T: temperature
    - patch_size: size of patches to process
    - stride: stride between patch centers
    - h: external field
\    
    Returns: [B, L, L] batch of relaxed spin configurations
    """
    device = spins.device
    if spins.dim() != 3:
        raise ValueError("spins must be a 3D tensor [B, L, L].")
    if spins.shape[1] != spins.shape[2]:
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
                rows = torch.arange(i - 1, i + patch_size + 1, device=device) % L
                cols = torch.arange(j - 1, j + patch_size + 1, device=device) % L
                
                # Extract patches for all batches using advanced indexing
                patch_spins = spins[:, rows][:, :, cols].clone()
                
                # Process all patches in the batch at once
                updated_patches = glauber_batched_torch(
                    patch_spins, beta=beta, h=h, n_events=n_events
                )
                
                # Handle periodic boundary conditions when writing back
                inner_rows = torch.arange(i, i + patch_size, device=device) % L
                inner_cols = torch.arange(j, j + patch_size, device=device) % L
                spins[:, inner_rows[:, None], inner_cols] = updated_patches
    return spins
