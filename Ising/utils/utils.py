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

def cal_mag_susceptibility(M, L, T=2.27):
    """Calculate magnetic susceptibility from magnetization data.
       $\mathcal{X} = \frac{L^2}{T}(\langle M^2 \rangle - \langle |M| \rangle^2)$

    Args:
        M (np.ndarray): Array of magnetization values. 
        L (int): Linear size of the system.
        T (float): Temperature of the system.

    Returns:
        float: Calculated magnetic susceptibility.
    """

    M = np.abs(M)  
    mag_susceptibility = np.mean(M ** 2) - np.mean(M) ** 2
    mag_susceptibility = mag_susceptibility * L ** 2 / T 
    return mag_susceptibility


class PartialEvolutionScheme():

    def __init__(self, T, h):
        self.T = T
        self.h = h

    def generate(self, data, patch_L):
        """Generate partial evolution data.

        Args:
            data (torch.Tensor): Input data of shape [N, B, L, L].
            patch_L (int): Size of the box for patch extraction.

        Returns:
            $x_{1, I}$, $\delta t$, partial idx I 
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the generated data.
        """
        
        L = data.shape[2]
        data = data.to(torch.float32)
        assert L % patch_L == 0, "L must be a multiple of patch_L"
        d = int(L // patch_L)

        X1_train_partial = [] 
        idx_train_partial = []
        time_step_train = []
        for i in tqdm(range(data.shape[0])):
            X0 = data[i] # [B, L, L]

            X0 = X0.reshape(-1, d, patch_L, d, patch_L) # [B, 2, 16, 2, 16]
            X0 = X0.permute(0, 1, 3, 2, 4) # [B, 2, 2, 16, 16]
            X0 = X0.flatten(1, 2) # [B, 4, 16, 16]
            idx = torch.randint(0, d ** 2, (X0.shape[0],), device=X0.device)

            X0_partial = X0[torch.arange(X0.shape[0]), idx].clone() # [batch_size, 16, 16]

            # $x_{1, I}$
            X1_partial, glauber_time = glauber_continuous_Ising(X0_partial.clone(), 1.0 / self.T, self.h)

            X1_train_partial.append(X1_partial)
            idx_train_partial.append(idx)
            time_step_train.append(glauber_time)

        X1_train_partial = torch.stack(X1_train_partial) # [N, B, L, L]
        idx_train_partial = torch.stack(idx_train_partial) # [N, B]
        time_step_train = torch.stack(time_step_train) # [N, B]

        return X1_train_partial, time_step_train, idx_train_partial



def glauber_continuous_Ising(
    spin: torch.Tensor,
    beta: Optional[float] = None,
    h: float = 0.0,
    n_events: Optional[int] = None,
) -> torch.Tensor:
    """Batched continuous-time Glauber KMC on 2D patches for Ising model.
    
    Args:
    
        spin: [B, patch_L, patch_L], batch of 2D patches
        beta: inverse temperature
        h: external field
        n_events: number of Glauber events (default: patch_L*patch_L)

    Returns: 
        spin: [B, patch_L, patch_L] batch of modified 2D spin tensors 
        glauber_time: [B] tensor of Glauber times
    """

    B = spin.shape[0]
    patch_L = spin.shape[1]
    device = spin.device
    glauber_time = torch.zeros(B, device=device, dtype=torch.float32)
    if n_events is None:
        n_events = patch_L * patch_L

    # for _ in range(patch_L * patch_L):
    for _ in range(n_events):
        # Neighbor sum using periodic boundary conditions
        R = torch.roll(spin, 1, dims=1) + torch.roll(spin, -1, dims=1) + \
            torch.roll(spin, 1, dims=2) + torch.roll(spin, -1, dims=2)

        dH = 2 * spin * R + 2 * h * spin  # Energy change
        rates = 1.0 / (1.0 + torch.exp(beta * dH))  # [B, L, L]

        rates_flat = rates.reshape(B, -1)  # [B, patch_L*patch_L]
        total_rates = rates_flat.sum(dim=1, keepdim=True)  # [B, 1]
        
        if torch.any(total_rates < 1e-10):
            raise ValueError("Total rate is too small, please check the input parameters.")

        # Normalize probabilities
        prob = rates_flat / total_rates  # [B, patch_L*patch_L]
        cum_probs = torch.cumsum(prob, dim=1)  # [B, patch_L*patch_L]

        # Generate random numbers for each batch element
        r = torch.rand(B, 1, device=device)
        selected_flat_idx = torch.searchsorted(cum_probs, r, right=False)
        selected_flat_idx = torch.clamp(selected_flat_idx, max=patch_L * patch_L - 1).squeeze(1)

        # Convert flat index to 2D indices
        x = selected_flat_idx // patch_L
        y = selected_flat_idx % patch_L

        # Flip the selected spin for each batch
        spin[torch.arange(B), x, y] *= -1

        # Update time using exponential waiting times
        rand_vals = torch.rand(B, device=spin.device)
        rand_vals = torch.clamp(rand_vals, min=1e-10, max=1.0)  # Avoid log(0)
        jumping_time = - torch.log(rand_vals) / total_rates.squeeze(1)

        glauber_time += jumping_time

    if torch.any(glauber_time == float('inf')):
        raise ValueError("glauber_time contains inf values, please check the input parameters.")

    return spin, glauber_time
    

def UpSample(spins: torch.Tensor, repeat: int) -> torch.Tensor:
    """2D version of UpSample

    Args:
        spins: [B, L, L] tensor of spins
        repeat: int, upsampling factor

    Returns:
        up: [B, L*repeat, L*repeat] tensor of upsampled spins
    """
    
    # Create repeatxrepeat duplication using repeat_interleave
    # Duplicate along x and y dimensions
    up = spins.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2)
    
    return up


def LocalRelax(
    spins: torch.Tensor,
    T: Optional[float] = None,
    patch_size: int = 16,
    stride: int = 8,
    h: float = 0.0,
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
    B, L, _ = spins.shape
    assert L % stride == 0, "L must be a multiple of stride"

    starts = range(0, L, stride)
    beta = 1.0 / T
    if n_events is None:
        n_events = patch_size * patch_size

    for i in starts:
        for j in starts:
            x_indices = torch.arange(i, i + patch_size, device=device) % L
            y_indices = torch.arange(j, j + patch_size, device=device) % L

            # Extract patches
            patch_spins = spins[:, x_indices[:, None], y_indices[None, :]]      

            # Perform Glauber updates on the extracted patches
            updated_patches, _ = glauber_continuous_Ising(
                patch_spins, beta=beta, h=h, n_events=n_events
            )

            spins[:, x_indices[:, None], y_indices[None, :]] = updated_patches
    return spins

