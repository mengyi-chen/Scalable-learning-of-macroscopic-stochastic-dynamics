import torch
import numpy as np
from scipy.interpolate import CubicSpline
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Upscale PDE trajectory data')
parser.add_argument('--input', type=str, default='../raw_data/X0_grid_100.pt', help='Input file path')
parser.add_argument('--output', type=str, default='X0_train_grid_200_upscaled.pt', help='Output file path')
parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for upscaling')
parser.add_argument('--method', type=str, default='linear', choices=['cubic_spline', 'linear'], help='Interpolation method')
parser.add_argument('--visualize', action='store_true', help='Create comparison visualization')
args = parser.parse_args()
print(args)

def UpSample(input_file, output_file, scale_factor=2, method='cubic_spline'):
    """
    Upscale PDE trajectory data using interpolation.
    
    Args:
        input_file (str): Path to input .pt file with coarse trajectories
        output_file (str): Path to save upscaled trajectories
        scale_factor (int): Factor by which to increase spatial resolution
        method (str): Interpolation method ('cubic_spline', 'linear')
    
    Returns:
        upscaled_data: Tensor with upscaled trajectories
    """
    
    # Load the coarse scale data
    print(f"Loading data from {input_file}...")
    data = torch.load(input_file, weights_only=False, map_location=torch.device('cpu'))
    
    n_traj, n_time, n_vars, n_grid_coarse = data.shape
    n_grid_fine = n_grid_coarse * scale_factor
    
    print(f"Input shape: {data.shape}")
    print(f"Output shape will be: [{n_traj}, {n_time}, {n_vars}, {n_grid_fine}]")
    
    # Create coordinate arrays using the same grid generation as the original solver
    # For coarse grid (original)
    dx_coarse = 1 / n_grid_coarse
    x_coarse = np.linspace(-0.5 * dx_coarse, dx_coarse * (n_grid_coarse + 0.5), n_grid_coarse + 2)[1:-1]
    
    # For fine grid (upscaled)  
    dx_fine = 1 / n_grid_fine
    x_fine = np.linspace(-0.5 * dx_fine, dx_fine * (n_grid_fine + 0.5), n_grid_fine + 2)[1:-1]
    
    print(f"Coarse grid: dx={dx_coarse:.6f}, x_range=[{x_coarse[0]:.6f}, {x_coarse[-1]:.6f}]")
    print(f"Fine grid: dx={dx_fine:.6f}, x_range=[{x_fine[0]:.6f}, {x_fine[-1]:.6f}]")
    
    # Initialize output tensor
    upscaled_data = torch.zeros(n_traj, n_time, n_vars, n_grid_fine, dtype=data.dtype)
    
    print("Upscaling trajectories...")
    
    # Process each trajectory
    for traj_idx in tqdm(range(n_traj), desc="Processing trajectories"):
        # Process each time step
        for time_idx in range(n_time):
            # Process each variable (u and v)
            for var_idx in range(n_vars):
                # Get the coarse data for this trajectory, time, and variable
                y_coarse = data[traj_idx, time_idx, var_idx, :].numpy()
                
                if method == 'cubic_spline':
                    # Create cubic spline interpolator
                    cs = CubicSpline(x_coarse, y_coarse, bc_type='natural')
                    # Interpolate to fine grid
                    y_fine = cs(x_fine)
                    
                elif method == 'linear':
                    # Linear interpolation
                    y_fine = np.interp(x_fine, x_coarse, y_coarse)
                    
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
                
                # Store the upscaled data
                upscaled_data[traj_idx, time_idx, var_idx, :] = torch.from_numpy(y_fine)
    
    print(f"Saving upscaled data to {output_file}...")
    torch.save(upscaled_data, output_file)
    
    # Print some statistics
    print("\nUpscaling completed!")
    print(f"Original data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Upscaled data range: [{upscaled_data.min():.4f}, {upscaled_data.max():.4f}]")
    
    return upscaled_data

def visualize_comparison(original_data, upscaled_data, traj_idx=0, time_idx=0, save_path=None):
    """
    Visualize comparison between original and upscaled data for a specific trajectory and time.
    
    Args:
        original_data: Original coarse data tensor
        upscaled_data: Upscaled fine data tensor  
        traj_idx: Trajectory index to visualize
        time_idx: Time step index to visualize
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    n_grid_coarse = original_data.shape[-1]
    n_grid_fine = upscaled_data.shape[-1]
    
    # Use the same grid generation as the original solver
    dx_coarse = 1 / n_grid_coarse
    x_coarse = np.linspace(-0.5 * dx_coarse, dx_coarse * (n_grid_coarse + 0.5), n_grid_coarse + 2)[1:-1]
    
    dx_fine = 1 / n_grid_fine  
    x_fine = np.linspace(-0.5 * dx_fine, dx_fine * (n_grid_fine + 0.5), n_grid_fine + 2)[1:-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot prey (u)
    ax1.plot(x_coarse, original_data[traj_idx, time_idx, 0, :].numpy(), 'o-', label=f'Original ({n_grid_coarse} points)', markersize=4)
    ax1.plot(x_fine, upscaled_data[traj_idx, time_idx, 0, :].numpy(), '-', label=f'Upscaled ({n_grid_fine} points)', alpha=0.8)
    ax1.set_title('Prey Population (u)')
    ax1.set_xlabel('Spatial coordinate')
    ax1.set_ylabel('Population density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot predator (v)
    ax2.plot(x_coarse, original_data[traj_idx, time_idx, 1, :].numpy(), 'o-', label=f'Original ({n_grid_coarse} points)', markersize=4)
    ax2.plot(x_fine, upscaled_data[traj_idx, time_idx, 1, :].numpy(), '-', label=f'Upscaled ({n_grid_fine} points)', alpha=0.8)
    ax2.set_title('Predator Population (v)')
    ax2.set_xlabel('Spatial coordinate')
    ax2.set_ylabel('Population density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Perform upscaling
    upscaled_data = UpSample(args.input, args.output, args.scale_factor, args.method)

    print("Creating visualization...")
    original_data = torch.load(args.input, weights_only=False, map_location=torch.device('cpu'))
    visualize_comparison(original_data, upscaled_data, save_path='upscaling_comparison.png')
