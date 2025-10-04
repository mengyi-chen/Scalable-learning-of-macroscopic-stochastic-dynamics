from logging import config
import os, sys
sys.path.append('../')
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import set_seed
import torch.nn.functional as F
from utils.utils import UpSample, LocalRelax, PartialEvolutionScheme
import warnings
warnings.filterwarnings("ignore")
import yaml

# Load parameters from YAML configuration file
with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_L', type=int, default=params['patch_L'], help="Length of the small lattice: L")
    parser.add_argument('--L', type=int, default=64, help="Length of the large lattice: L")
    parser.add_argument('--N_target_train', type=int, default=params['N_target_train'], help="number of target data")
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], help="number of validation data")
    parser.add_argument('--T', type=float, default=params['T'], help="Temperature")
    parser.add_argument('--h', type=float, default=params['h'], help="externel field strength")
    parser.add_argument('--seed', type=int, default=params['seed'], help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=params['steps'], help="Number of glauber dynamics steps")
    parser.add_argument('--gpu_idx', type=int, default=params['gpu_idx'], help="GPU index to use")
    parser.add_argument('--n_events', type=int, default=None, help="Number of events for Glauber dynamics")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ========= parse args and set seed =========
    args = args_parser()
    print("> Settings: ", args)

    # Set all random seeds for reproducibility
    set_seed(args.seed)

    # ========= Load the configuration of small system =========
    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    loadpath = f'../raw_data/L{args.patch_L}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    X_train_small = torch.load(os.path.join(loadpath, 'X0.pt'), weights_only=True, map_location=device) # [n_tra, len_per_tra, patch_L, patch_L]
    X_train_small = X_train_small.flatten(0, 1) # [N, patch_L, patch_L]
    print('Number of snapshots of small configurations:', X_train_small.shape[0])
    rootpath = f'./scaleup_patch_L_{args.patch_L}_L{args.L}_h{args.h}_T{args.T:.2f}'
    os.makedirs(rootpath, exist_ok=True)

    # ========= Hierarchical Upsampling Scheme =========
    print('='*20, 'Hierarchical Upsampling Scheme', '='*20)
    assert args.patch_L == 16, "Currently only support patch_L = 16"
    assert args.L in [16, 32, 48, 64, 128], "Currently only support L = 16,32, 48, 64, 128"
    print(f'Generating large system configurations (n={args.L}^2) from small system configurations (n_s={args.patch_L}^2) ...')
    n_iters = np.log2(args.L // args.patch_L).astype(int) # log2(L // patch_L)
    if args.L == 48:
        repeat = 3
    else:
        repeat = 2
    for iter in range(n_iters):
        print('iteration:', iter)
        print('small system shape:', X_train_small.shape)

        X0_train = []
        for _ in tqdm(range(args.N_target_train // args.batch_size)):
            idx = torch.randint(0, X_train_small.shape[0], (args.batch_size,), device=device)

            config_small = X_train_small[idx] # [batch_size, patch_L, patch_L]
            config_large = UpSample(config_small, repeat=repeat) # [batch_size, L, L, L]
            if iter == n_iters - 1 and args.n_events is not None:
                config_large = LocalRelax(config_large, args.T, h=args.h, n_events=args.n_events) # [batch_size, L, L]
            else:
                config_large = LocalRelax(config_large, args.T, h=args.h, n_events=None) # [batch_size, L, L]
            X0_train.append(config_large.detach().cpu())
            
        X0_train = torch.stack(X0_train) # [N, batch_size, L, L]
        print('large system shape:', X0_train.shape)
        
        if iter < n_iters - 1:
            X_train_small = X0_train.flatten(0, 1).to(device) # [N * batch_size, L, L]
    
    if args.patch_L == args.L:

        idx = np.random.choice(X_train_small.shape[0], args.N_target_train, replace=False)
        X0_train = X_train_small[idx]
        X0_train = X0_train.reshape(-1, args.batch_size, args.L, args.L) # [N, B, L, L]
    
    X_train_large = X0_train
    torch.save(X_train_large.detach().cpu(), os.path.join(rootpath, 'X0_train.pt'))

    # # ========= Partial Evolution Scheme =========
    print('='*20, 'Partial Evolution Scheme', '='*20)
    print('Final training data shape:', X0_train.shape) # [N, B, L, L]
    
    generator = PartialEvolutionScheme(args.T, args.h)
    X1_train_partial, time_step_train, idx_train_partial = generator.generate(X0_train.to(device), args.patch_L)
    
    print('X1_train_partial shape:', X1_train_partial.shape)
    print('idx_train_partial shape:', idx_train_partial.shape)
    print('time_step_train shape:', time_step_train.shape)

    torch.save(X1_train_partial, os.path.join(rootpath, 'X1_train_partial.pt'))
    torch.save(idx_train_partial, os.path.join(rootpath, 'idx_train_partial.pt'))
    torch.save(time_step_train, os.path.join(rootpath, 'time_step_train.pt'))
