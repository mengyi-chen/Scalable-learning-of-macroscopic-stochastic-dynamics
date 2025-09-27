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
# from glauber2d import Glauber2DIsing
import seaborn as sns
from sklearn.model_selection import train_test_split

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=64, help="Length of the lattice: L")
    parser.add_argument('--T', type=float, default=2.27, help="Temperature")
    parser.add_argument('--h', type=float, default=0.0, help="externel field strength")
    parser.add_argument('--steps', type=int, default=32000, help="Number of glauber dynamics steps")
    # parser.add_argument('--dt', type=float, default=0.1, help="Time step of glauber dynamics")
    parser.add_argument('--save_interval', type=int, default=10, help="Interval of saving the microscopic configuration")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--num_run', type=int, default=50, help='')
    parser.add_argument('--n_proc', type=int, default=50, help="Number of processors for multiprocessing")
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
    rootpath = f'./L{args.L}_MC{args.steps}_h{args.h}_T{args.T:.2f}'
    if not os.path.exists(rootpath):  
        os.makedirs(rootpath)

    config_time = np.array(config_time, dtype=np.int8)
    config_time = torch.tensor(config_time, dtype=torch.int8)
    print('config_time shape:', config_time.shape)

    X0 = config_time

    X0_train, X0_val = train_test_split(X0, test_size=0.2, random_state=args.seed)
    print('X0_train shape:', X0_train.shape)
    print('X0_val shape:', X0_val.shape)

    if args.L == 16:
        torch.save(X0_train, os.path.join(rootpath, 'X0_train.pt')) 
    torch.save(X0_val, os.path.join(rootpath, 'X0_val.pt'))
    
    # ========= plot =========
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
    # sns.histplot(M, bins=50, kde=True, stat="density", color='orange', label='Histogram', alpha=0.5)
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
