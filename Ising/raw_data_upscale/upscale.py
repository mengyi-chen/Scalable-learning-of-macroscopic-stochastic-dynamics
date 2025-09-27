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
from utils.utils import scaleup_batched_torch, patchwise_relax_batched_torch, glauber_continuous, PartialGenerator
import warnings
warnings.filterwarnings("ignore")

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L_small', type=int, default=16, help="Length of the small lattice: L")
    parser.add_argument('--L_large', type=int, default=64, help="Length of the large lattice: L")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patches: patch_size")
    # parser.add_argument('--box_L', type=int, nargs='+', default=[8, 16, 32, 64], help="Length of the box: box_L (can be a list)")
    # parser.add_argument('--box_L', type=int, nargs='+', default=[64], help="Length of the box: box_L (can be a list)")
    parser.add_argument('--n_events', type=int, default=None, help="Number of events for Glauber dynamics")
    # parser.add_argument('--n_events', type=int, nargs='+', default=[None], help="Number of events for Glauber dynamics")
    parser.add_argument('--N_target_train', type=int, default=100000, help="number of target data")
    parser.add_argument('--N_target_val', type=int, default=40000, help="number of validation data")
    parser.add_argument('--batch_size', type=int, default=2000, help="number of validation data")
    parser.add_argument('--T', type=float, default=2.5, help="Temperature")
    parser.add_argument('--h', type=float, default=0.1, help="externel field strength")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=200, help="Number of glauber dynamics steps")
    parser.add_argument('--gpu_idx', type=int, default=5, help="GPU index to use")
    parser.add_argument('--relax_n_events', type=int, default=None, help="Number of events for relaxation")
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
    loadpath = f'../raw_data/L{args.L_small}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    X_train_small = torch.load(os.path.join(loadpath, 'X0.pt'), weights_only=True, map_location=device) # [n_tra, len_per_tra, L, L, L]
    X_train_small = X_train_small.flatten(0, 1) # [N, L_small, L_small, L_small]
    print('Number of snapshots of small configurations:', X_train_small.shape[0])
    rootpath = f'./scaleup_box_L_{args.L_small}_L{args.L_large}_h{args.h}_T{args.T:.2f}'
    os.makedirs(rootpath, exist_ok=True)

    # ========= Generate the configuration of large system =========
    assert args.L_small in [8, 16, 32, 64], "L_small must be in [8, 16, 32, 64]"
    assert args.L_large == 64, "Currently only support L_large = 64"
    n_iters = np.log2(args.L_large // args.L_small).astype(int) # log2(L_large // L_small)
    for iter in range(n_iters):
        print('iteration:', iter)
        print('small system shape:', X_train_small.shape)

        X0_train = []
        for _ in tqdm(range(args.N_target_train // args.batch_size)):
            idx = torch.randint(0, X_train_small.shape[0], (args.batch_size,), device=device)

            config_small = X_train_small[idx] # [batch_size, L_small, L_small, L_small]
            config_large = scaleup_batched_torch(config_small, repeat=2) # [batch_size, L_large, L_large, L_large]
            config_large = patchwise_relax_batched_torch(config_large, args.T, h=args.h, n_events=args.relax_n_events)
            X0_train.append(config_large)

        if iter == n_iters - 1:
            for _ in tqdm(range(args.N_target_train // args.batch_size)):
                config_large = []
                for i in range(args.batch_size):
                    initial_mag = 2 * np.random.rand() - 1
                    num_up = int((1 + initial_mag) * args.L_large ** 2 / 2)
                    num_down = args.L_large ** 2 - num_up
                    spins = np.concatenate([np.ones(num_up), -np.ones(num_down)])
                    np.random.shuffle(spins)
                    config_large.append(spins.reshape(args.L_large, args.L_large))
                config_large = torch.tensor(config_large, device=device, dtype=torch.float32) # [batch_size, L_large, L_large, L_large]
                X0_train.append(config_large)
        
        X0_train = torch.stack(X0_train) # [N, batch_size, L_large, L_large, L_large]
        print('large system shape:', X0_train.shape)
        
        if iter < n_iters - 1:
            X_train_small = X0_train.flatten(0, 1) # [N * batch_size, L_large, L_large, L_large]

    if args.L_small == args.L_large:

        idx = np.random.choice(X_train_small.shape[0], args.N_target_train, replace=False)
        X0_train = X_train_small[idx]
        X0_train = X0_train.reshape(-1, args.batch_size, args.L_large, args.L_large) # [N, B, L_large, L_large]
    
    X_train_large = X0_train
    torch.save(X0_train, os.path.join(rootpath, 'X0_train.pt'))

    # ========= Generate the partial labels =========            
    print('Final training data shape:', X0_train.shape) # [N, B, L_large, L_large]

    generator = PartialGenerator(args.T, args.h)
    print('='*20, f'Processing box_L: {args.L_small}', '='*20)
    X1_train_partial, time_step_train, idx_train_partial = generator.generate(X0_train, args.L_small, n_events=args.n_events)
    
    print('X1_train_partial shape:', X1_train_partial.shape)
    print('idx_train_partial shape:', idx_train_partial.shape)
    print('time_step_train shape:', time_step_train.shape)

    torch.save(X1_train_partial, os.path.join(rootpath, 'X1_train_partial.pt'))
    torch.save(idx_train_partial, os.path.join(rootpath, 'idx_train_partial.pt'))
    torch.save(time_step_train, os.path.join(rootpath, 'time_step_train.pt'))
