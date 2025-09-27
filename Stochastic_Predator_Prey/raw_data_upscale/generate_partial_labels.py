import os, sys
sys.path.append('../')
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import *
import torch.nn.functional as F
import yaml

with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', default=params['a'], type=float, help='parameter a')
    parser.add_argument('--b', default=params['b'], type=float, help='parameter b')
    parser.add_argument('--D', default=params['D'], type=float, help='parameter D')
    parser.add_argument('--dt', default=params['dt'], type=float, help='dt')
    parser.add_argument('--noise_level', default=params['noise_level'], type=float, help='noise level')
    parser.add_argument('--solver', default=params['solver'], type=str, choices=['euler', 'RK4'])
    parser.add_argument('--seed', default=params['seed'], type=int, help='Random seed')
    parser.add_argument('--n_parts', default=params['n_parts'], type=int, help='Number of parts to divide the grid into')
    parser.add_argument('--gpu_idx', default=1, type=int, help='GPU index')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # ========= parse args and set seed =========
    args = args_parser()
    print("> Settings: ", args)

    # Set all random seeds for reproducibility
    set_seed(args.seed)

    # ========= save the microscopic configuration data =========
    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    X0 = torch.load('./X0_train_grid_200_upscaled.pt', weights_only=True, map_location=device)
    model = NetPartial(args.dt, device, a=args.a, b=args.b, D=args.D).to(device)
    print('X0 shape:', X0.shape)

    # ========= generate X1_partial data =========
    X1_partial = []
    idx_train_partial = []
    assert X0.shape[-1] % args.n_parts == 0, "Grid size must be divisible by n_parts"
    part_size = X0.shape[-1] // args.n_parts
    for i in tqdm(range(X0.shape[0])):
        x0 = X0[i].clone() # [B, 2, nx]    
        B = x0.shape[0]
        
        # Create full ghost grid first: [B, 2, nx] -> [B, 2, nx + 2*n_parts]
        # Each part gets one ghost cell on each side
        x0_ghost = torch.zeros(B, 2, x0.shape[-1] + 2 * args.n_parts, device=x0.device)
        
        # Fill the ghost grid part by part with ghost cells
        for part_idx in range(args.n_parts):
            start_orig = part_idx * part_size
            end_orig = (part_idx + 1) * part_size
            start_ghost = part_idx * (part_size + 2) + 1  # +1 for left ghost cell
            end_ghost = start_ghost + part_size
            
            # Copy the original data
            x0_ghost[:, :, start_ghost:end_ghost] = x0[:, :, start_orig:end_orig]
            
            # Add left ghost cell
            if part_idx == 0:
                # First part: use Neumann boundary (replicate first value)
                x0_ghost[:, :, start_ghost - 1] = x0[:, :, start_orig]
            else:
                # Use last value from previous part
                x0_ghost[:, :, start_ghost - 1] = x0[:, :, start_orig - 1]
            
            # Add right ghost cell
            if part_idx == args.n_parts - 1:
                # Last part: use Neumann boundary (replicate last value)
                x0_ghost[:, :, end_ghost] = x0[:, :, end_orig - 1]
            else:
                # Use first value from next part
                x0_ghost[:, :, end_ghost] = x0[:, :, end_orig]
        
        # Reshape to [B, 2, n_parts, part_size + 2] for easy slicing
        x0_ghost = x0_ghost.view(B, 2, args.n_parts, part_size + 2)
        
        # Randomly select which part to use for each batch
        idx_partial = torch.randint(0, args.n_parts, (B,), device=x0.device)

        # Extract the selected partial ghost data
        x0_partial_ghost = x0_ghost[torch.arange(B), :, idx_partial]  # [B, 2, part_size + 2]

        if args.solver == 'euler':
            _, x1_partial = euler_partial(model, x0_partial_ghost, args.dt, noise_level=args.noise_level)
        elif args.solver == 'RK4':
            _, x1_partial = rk4_partial(model, x0_partial_ghost, args.dt, noise_level=args.noise_level)

        idx_train_partial.append(idx_partial)
        X1_partial.append(x1_partial)
        
    X1_partial = torch.stack(X1_partial)
    idx_train_partial = torch.stack(idx_train_partial)

    print('X1_partial shape:', X1_partial.shape)
    print('idx_train_partial shape:', idx_train_partial.shape)
        
    # Save the partial data
    torch.save(idx_train_partial, './idx_train_partial.pt')
    torch.save(X1_partial, f'./X1_train_partial.pt')
    
    print("Partial data saved successfully!")
    