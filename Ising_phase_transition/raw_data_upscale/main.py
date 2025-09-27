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
from utils.utils import glauber_continuous
from upscale_torch import scaleup_batched_torch, patchwise_relax_batched_torch
import warnings
warnings.filterwarnings("ignore")

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L_small', type=int, default=64, help="Length of the small lattice: L")
    parser.add_argument('--L_large', type=int, default=64, help="Length of the large lattice: L")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patches: patch_size")
    parser.add_argument('--box_L', type=int, default=16, help="Length of the box: box_L")
    parser.add_argument('--N_target_train', type=int, default=400000, help="number of target data")
    parser.add_argument('--N_target_val', type=int, default=40000, help="number of validation data")
    parser.add_argument('--batch_size', type=int, default=5000, help="number of validation data")
    parser.add_argument('--T', type=float, default=2.27, help="Temperature")
    parser.add_argument('--h', type=float, default=0.0, help="externel field strength")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=32000, help="Number of glauber dynamics steps")
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

    # ========= Load the configuration of small system =========
    if args.L_small == 16:
        loadpath = f'../raw_data/L{args.L_small}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    else:
        loadpath = f'./scaleup_L{args.L_small}_h{args.h}_T{args.T:.2f}'


    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    X_train_small = torch.load(os.path.join(loadpath, 'X0_train.pt'), weights_only=True, map_location=device) # [n_tra, len_per_tra, L, L]
    # X_val_small = torch.load(os.path.join(loadpath, 'X0_val.pt'), weights_only=True, map_location=device) # [n_tra_val, len_per_tra, L, L]

    X_train_small = X_train_small.flatten(0, 1) # [N, L_small, L_small]
    # X_val = X_val_small.flatten(0, 1) # [N, L_small, L_small]

    # ========= Generate the configuration of large system =========

    assert args.L_large % args.L_small == 0, "L_large must be a multiple of L_small"
    repeat = args.L_large // args.L_small

    X0_train = []
    X0_val = []
    for _ in tqdm(range(args.N_target_train // args.batch_size)):
        idx = torch.randint(0, X_train_small.shape[0], (args.batch_size,), device=device)

        config_small = X_train_small[idx] # [batch_size, L_small, L_small]
        config_large = scaleup_batched_torch(config_small, repeat=repeat) # [batch_size, L_large, L_large]

        config_large = patchwise_relax_batched_torch(config_large, args.T, h=args.h, n_events=args.n_events)
        X0_train.append(config_large)

    X0_train = torch.stack(X0_train) # [N_target_train, L_large, L_large]
    print('X0_train shape:', X0_train.shape)

    # for _ in tqdm(range(args.N_target_val // args.batch_size)):
    #     idx = torch.randint(0, X_val.shape[0], (args.batch_size,), device=device)
    #     config_small = X_val[idx] # [batch_size, L, L]
    #     config_large = scaleup_batched_torch(config_small)     
    #     config_large = patchwise_relax_batched_torch(config_large, args.T, h=args.h)
    #     X0_val.append(config_large)
    # X0_val = torch.stack(X0_val) # [N_target_val, L*2, L*2]
    # print('X0_val shape:', X0_val.shape)

    rootpath = f'./scaleup_L{args.L_large}_h{args.h}_T{args.T:.2f}'
    os.makedirs(rootpath, exist_ok=True)
    torch.save(X0_train, os.path.join(rootpath, 'X0_train.pt'))
    # torch.save(X0_val , os.path.join(rootpath, 'X0_val.pt'))

    # ========= Generate the partial labels =========

    print('X0_train shape:', X0_train.shape)
    # print('X0_val shape:', X0_val.shape)


    d = int(args.L_large // args.box_L)

    X1_train_partial = []
    time_step_train = []
    idx_partial_train = []
    for i in tqdm(range(X0_train.shape[0])):
        tra = X0_train[i] # [3200, 64, 64]

        tra = tra.reshape(-1, d, args.box_L, d, args.box_L)
        tra = tra.permute(0, 1, 3, 2, 4) # [3200, 2, 2, 32, 32]
        tra = tra.flatten(1, 2) # [3200, 4, 32, 32]
        idx = torch.randint(0, d ** 2, (tra.shape[0],), device=tra.device)
        
        tra_partial = tra[torch.arange(tra.shape[0]), idx] # [3200, 32, 32]
        tra_dt_partial, glauber_time = glauber_continuous(tra_partial, args.box_L, 1.0 / args.T, args.h)

        X1_train_partial.append(tra_dt_partial)
        time_step_train.append(glauber_time)
        idx_partial_train.append(idx)

    X1_train_partial = torch.stack(X1_train_partial)
    time_step_train = torch.stack(time_step_train)
    idx_partial_train = torch.stack(idx_partial_train)
    print('X1_train_partial shape:', X1_train_partial.shape)
    print('time_step_train shape:', time_step_train.shape)
    print('idx_partial_train shape:', idx_partial_train.shape)

    torch.save(X1_train_partial, os.path.join(rootpath, f'X1_train_partial.pt'))
    torch.save(time_step_train, os.path.join(rootpath, f'time_step_train_partial.pt'))
    torch.save(idx_partial_train, os.path.join(rootpath, f'idx_partial_train.pt'))

    # ========= save val data =========
    # X1_val = []
    # time_step_val = []
    # for i in tqdm(range(X0_val.shape[0])):
    #     tra = X0_val[i] # [B, L, L]
    #     tra_dt, glauber_time = glauber_continuous(tra, args.L_large, 1.0 / args.T, args.h)
    #     X1_val.append(tra_dt)
    #     time_step_val.append(glauber_time)

    # X1_val = torch.stack(X1_val)
    # time_step_val = torch.stack(time_step_val)

    # print('X1_val shape:', X1_val.shape)
    # print('time_step_val shape:', time_step_val.shape)

    # torch.save(X1_val, os.path.join(rootpath, f'X1_val.pt'))
    # torch.save(time_step_val, os.path.join(rootpath, f'time_step_val.pt'))