from logging import config
import os
import time
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from glauber_curie_weiss import Glauber2DCW

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=16, help="Length of the lattice: L")
    parser.add_argument('--T', type=float, default=1.1, help="Temperature")
    parser.add_argument('--h', type=float, default=0.1, help="externel field strength")
    parser.add_argument('--steps', type=int, default=1000, help="Number of glauber dynamics steps")
    # parser.add_argument('--dt', type=float, default=0.01, help="Time step of glauber dynamics")
    # parser.add_argument('--save_interval', type=int, default=1, help="Interval of saving the microscopic configuration")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--num_run', type=int, default=100, help='')
    parser.add_argument('--n_proc', type=int, default=100, help="Number of processors for multiprocessing")
    parser.add_argument('--mag', type=float, default=None, help="Target magnetization")
    parser.add_argument('--n_events', type=int, default=None, help="Number of events for Glauber dynamics")
    parser.add_argument('--flag_test', action='store_true', help="Flag for test mode")
    args = parser.parse_args()
    return args

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# Monte Carlo task
def mcmc_task(args, Ts, seed_list):

    Es, Ms, Cs, Xs, Ms_time, configs_time, kmc_times = [], [], [], [], [], [], []

    m = Glauber2DCW(args)
    pbar = tqdm(desc="Progress: ".format(id), total=len(Ts))
    for T, seed in zip(Ts, seed_list):
        # Set seeds for each individual run to ensure reproducibility
        np.random.seed(int(seed))
        random.seed(int(seed))

        E, M, C, X, M_time, config_time, kmc_time = m.simulate(1 / T, h=args.h, n_events=args.n_events)
        Es.append(E)
        Ms.append(abs(M))
        Cs.append(C)
        Xs.append(X)
        Ms_time.append(M_time)
        configs_time.append(config_time)
        kmc_times.append(kmc_time)
        pbar.update(1)
    return Es, Ms, Cs, Xs, Ms_time, configs_time, kmc_times


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
        Es_list, Ms_list, Cs_list, Xs_list, Ms_time_list, configs_time_list, kmc_times_list = zip(*pool.starmap(mcmc_task, zip(args_list, Ts_list, seeds_list)))
    Es, Ms, Cs, Xs, Ms_time, = sum(Es_list, []), sum(Ms_list, []), sum(Cs_list, []), sum(Xs_list, []), sum(Ms_time_list, [])
    config_time = sum(configs_time_list, [])
    kmc_times = sum(kmc_times_list, [])
    kmc_times = np.array(kmc_times, dtype=np.float32)
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

    X0 = config_time
        
    if args.flag_test:
        torch.save(X0, os.path.join(rootpath, 'X0_val.pt'))
    else:
        torch.save(X0, os.path.join(rootpath, 'X0.pt')) 
    np.save(os.path.join(rootpath, 'kmc_times.npy'), kmc_times)

    # ========= plot =========
    Ms_time = np.array(Ms_time, dtype=np.float32)
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(1, 1, 1)
    for i in range(Ms_time.shape[0]):
        plt.plot(kmc_times[i], Ms_time[i])
        if i > 50:
            break
    plt.xlabel("time (t)", fontsize=20)
    plt.ylabel('Magnetization', fontsize=20)
    axes.set_ylim(-1, 1)
    
    plot_name = rootpath + f'/glauber_time.png'
    plt.savefig(plot_name)
    plt.close()
    print("Saved the plot into {}.".format(plot_name))