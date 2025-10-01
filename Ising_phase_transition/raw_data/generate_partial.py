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
    parser.add_argument('--L', type=int, default=128, help="Length of the lattice: L")
    parser.add_argument('--patch_L', type=int, default=16, help="Length of the box: patch_L")
    parser.add_argument('--T', type=float, default=2.27, help="Temperature")
    parser.add_argument('--h', type=float, default=0.0, help="externel field strength")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=32000, help="Number of glauber dynamics steps")
    parser.add_argument('--gpu_idx', type=int, default=5, help="GPU index to use")
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

    d = int(args.L // args.patch_L)  

    if args.L == 16:
        X0_train = torch.load(os.path.join(rootpath, 'X0_train.pt'), weights_only=True, map_location=device) 
        X1_train = []
        time_step_train = []
        for i in tqdm(range(X0_train.shape[0])):
            tra = X0_train[i] # [3200, 16, 16]
            tra_dt, glauber_time = glauber_continuous(tra, args.patch_L, 1.0 / args.T, args.h)

            X1_train.append(tra_dt)
            time_step_train.append(glauber_time)

        X1_train = torch.stack(X1_train)
        time_step_train = torch.stack(time_step_train)
        print('X1_train shape:', X1_train.shape)
        print('time_step_train shape:', time_step_train.shape)

        torch.save(X1_train, os.path.join(rootpath, f'X1_train.pt'))
        torch.save(time_step_train, os.path.join(rootpath, f'time_step_train.pt'))

    # ========= save val data =========
    X1_val = []
    time_step_val = []
    for i in tqdm(range(X0_val.shape[0])):
        tra = X0_val[i] # [B, L, L]
        tra_dt, glauber_time = glauber_continuous(tra, args.L, 1.0 / args.T, args.h)
        X1_val.append(tra_dt)
        time_step_val.append(glauber_time)

    X1_val = torch.stack(X1_val)
    time_step_val = torch.stack(time_step_val)

    torch.save(X1_val, os.path.join(rootpath, f'X1_val.pt'))
    torch.save(time_step_val, os.path.join(rootpath, f'time_step_val.pt'))