import os, sys
sys.path.append('../')
from utils.utils import cal_mag_susceptibility
import time
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from contextlib import contextmanager
from glauber2d_ising import Glauber2DIsing
import seaborn as sns
import yaml
from utils.utils import glauber_continuous_Ising

# Load parameters from YAML configuration file
with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=params['L'], help="Length of the lattice: L")
    parser.add_argument('--T', type=float, default=params['T'], help="Temperature")
    parser.add_argument('--h', type=float, default=params['h'], help="externel field strength")
    parser.add_argument('--steps', type=int, default=params['steps'], help="Number of glauber dynamics steps")
    parser.add_argument('--seed', type=int, default=params['seed'], help='Random seed (default: 0)')
    parser.add_argument('--num_run', type=int, default=params['num_run'], help='Number of independent runs (default: 100)')
    parser.add_argument('--n_proc', type=int, default=params['n_proc'], help="Number of processors for multiprocessing")
    parser.add_argument('--mag', type=float, default=None, help="Target magnetization")
    parser.add_argument('--data_mode', type=str, choices=['train', 'val', 'test'], default='train', help="Mode of the data: train, val, or test")
    # parser.add_argument('--glauber_dynamics', type=str, choices=['continuous', 'discrete'], default='continuous', help="Type of Glauber dynamics")
    parser.add_argument('--gpu_idx', type=int, default=params['gpu_idx'], help="GPU index to use")
    parser.add_argument('--save_interval', type=int, default=params['save_interval'], help="Interval of saving the microscopic configuration")
    args = parser.parse_args()
    return args

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# Monte Carlo task
def mcmc_task(args, Ts, seed_list):

    Es, Ms, Cs, Xs, Ms_time, configs_time = [], [], [], [], [], []

    m = Glauber2DIsing(args)
    pbar = tqdm(desc="Progress: ".format(id), total=len(Ts))
    for T, seed in zip(Ts, seed_list):
        # Set seeds for each individual run to ensure reproducibility
        np.random.seed(int(seed))
        random.seed(int(seed))

        E, M, C, X, M_time, config_time = m.simulate(1 / T, h=args.h)
        Es.append(E)
        Ms.append(abs(M))
        Cs.append(C)
        Xs.append(X)
        Ms_time.append(M_time)
        index = list(np.arange(1000)) + list(range(1000, args.steps-1, args.save_interval))
        configs_time.append(config_time[index])
        pbar.update(1)
    return Es, Ms, Cs, Xs, Ms_time, configs_time


if __name__ == "__main__":
    # ========= parse args and set seed =========
    args = args_parser()
    print("> Settings: ", args)
    
    n_proc = mp.cpu_count() if args.n_proc == 0 else args.n_proc
    print("> Number of processes: ", n_proc)
    
    # Set all random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ========= Monte Carlo in a pool =========
    start = time.time()
    args_list = [args for _ in range(n_proc)]
    Ts = [args.T for _ in range(args.num_run)]
    seeds = [i for i in range(args.num_run)]
    Ts_list = np.array_split(np.array(Ts), n_proc) 
    seeds_list = np.array_split(np.array(seeds), n_proc) 

    with poolcontext(processes=n_proc) as pool:
        Es_list, Ms_list, Cs_list, Xs_list, Ms_time_list, configs_time_list = zip(*pool.starmap(mcmc_task, zip(args_list, Ts_list, seeds_list)))
    Es, Ms, Cs, Xs, Ms_time, = sum(Es_list, []), sum(Ms_list, []), sum(Cs_list, []), sum(Xs_list, []), sum(Ms_time_list, [])
    config_time = sum(configs_time_list, [])
    print("\n> Elapsed time: {:4f}s".format(time.time() - start))
    print('mean of E:', np.mean(Es))
    print('mean of M:', np.mean(Ms))
    print('mean of C:', np.mean(Cs))
    print('mean of X:', np.mean(Xs))

    # ========= save the microscopic configuration data =========

    if args.mag is None:
        rootpath = f'./L{args.L}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    else:
        rootpath = f'./L{args.L}_MC{args.steps}_h{args.h}_T{args.T:.2f}_mag{args.mag}'

    if not os.path.exists(rootpath):  
        os.makedirs(rootpath)

    config_time = np.array(config_time, dtype=np.int8)
    config_time = torch.tensor(config_time, dtype=torch.int8)
    print('config_time shape:', config_time.shape)

    # ========= Visualization =========
    Ms_time = np.array(Ms_time, dtype=np.float32)
    fig = plt.figure(figsize=(12, 3))
    axes = fig.add_subplot(1, 1, 1)
    for i in range(Ms_time.shape[0]):
        plt.plot(Ms_time[i])
        if i > 10:
            break
    plt.xlabel("time (t)", fontsize=20)
    plt.ylabel('Magnetization', fontsize=20)
    axes.set_ylim(-1, 1)
    
    plt.axis('tight')
    plot_name = rootpath + f'/glauber_time.png'
    plt.savefig(plot_name)
    plt.close()
    print("Saved the plot into {}.".format(plot_name))

    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(1, 1, 1)

    print('Ms_time shape:' , Ms_time.shape)
    M = Ms_time[:, (args.steps // 2):].flatten()
    cal_mag_susceptibility = cal_mag_susceptibility(M, args.L)
    print(f"Magnetic susceptibility: {cal_mag_susceptibility:.4f}")

    M = M[::10]
    M = np.abs(M)  # Take absolute value of magnetization

    torch.save(torch.tensor(M, dtype=torch.float32), os.path.join(rootpath, 'M_equilibrium.pt'))
    np.save(os.path.join(rootpath, 'M_equilibrium.npy'), M)
    sns.kdeplot(M, fill=False, bw_adjust=2.5, cut=0)
    plt.xlim(-1, 1)  # Set x-axis limits for magnetization range
    plt.xlabel("Magnetization", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title(f"Magnetization distribution at T={args.T:.2f}", fontsize=24)
    plt.grid(True)
    plot_name = rootpath + f"/magnetization_distribution.png"
    print("Saved the plot into {}.".format(plot_name))
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()  # Close the figure to free memory

    # ========= save data =========
    X0 = config_time
    
    if args.data_mode == 'val':
        device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
        save_num_run = int(0.2 * args.num_run)
        X0 = X0[:save_num_run]
        X0_val = X0.to(device)
        X1_val = []
        time_step_val = []
        for i in tqdm(range(X0_val.shape[0])):
            tra = X0_val[i].clone() # [B, L, L]
            tra_dt, glauber_time = glauber_continuous_Ising(tra.clone(), 1.0 / args.T, args.h)
            X1_val.append(tra_dt)
            time_step_val.append(glauber_time)

        X1_val = torch.stack(X1_val)
        time_step_val = torch.stack(time_step_val)

        torch.save(X1_val.detach().cpu(), os.path.join(rootpath, f'X1_val.pt'))
        torch.save(time_step_val.detach().cpu(), os.path.join(rootpath, f'time_step_val.pt'))
    
    save_name = {'train': 'X0.pt', 'val': 'X0_val.pt', 'test': 'X0_test.pt'}
    torch.save(X0, os.path.join(rootpath, save_name[args.data_mode]))
