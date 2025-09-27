import os, sys
sys.path.append('../')
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import set_seed
import numba
from utils.utils import glauber_continuous
import torch.nn.functional as F

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=32, help="Length of the lattice: L")
    parser.add_argument('--T', type=float, default=1.1, help="Temperature")
    parser.add_argument('--h', type=float, default=0.1, help="externel field strength")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=200, help="Number of glauber dynamics steps")
    parser.add_argument('--gpu_idx', type=int, default=5, help="GPU index to use")
    parser.add_argument('--n_events', type=int, default=None, help="Number of events for Glauber dynamics")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # ========= parse args and set seed =========
    args = args_parser()
    print("> Settings: ", args)

    # Set all random seeds for reproducibility
    set_seed(args.seed)

    # ========= save the microscopic configuration data =========
    rootpath = f'./L{args.L}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    X0_val = torch.load(os.path.join(rootpath, 'X0_val.pt'), weights_only=True, map_location=device)

    # ========= save val data =========
    X1_val = []
    time_step_val = []
    for i in tqdm(range(X0_val.shape[0])):
        tra = X0_val[i].clone() # [B, L, L]
        tra_dt, glauber_time = glauber_continuous(tra.clone(), 1.0 / args.T, args.h, args.n_events)
        X1_val.append(tra_dt)
        time_step_val.append(glauber_time)

    X1_val = torch.stack(X1_val)
    time_step_val = torch.stack(time_step_val)

    torch.save(X1_val, os.path.join(rootpath, f'X1_val.pt'))
    torch.save(time_step_val, os.path.join(rootpath, f'time_step_val.pt'))

    # X1_val = torch.load(os.path.join(rootpath, 'X1_val.pt'), weights_only=True, map_location=device)

    z0_val = torch.mean(X0_val.to(torch.float32), dim=(2, 3))  # Changed from (2, 3, 4) to (2, 3) for 2D
    z1_val = torch.mean(X1_val.to(torch.float32), dim=(2, 3))  # Changed from (2, 3, 4) to (2, 3) for 2D
    print('z0_val shape:', z0_val.shape)
    print('z1_val shape:', z1_val.shape)

    torch.save(z0_val, os.path.join(rootpath, f'z0_val.pt'))   
    torch.save(z1_val, os.path.join(rootpath, f'z1_val.pt'))
