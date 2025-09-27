import os
import csv
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
    parser.add_argument('--init_temp', type=float, default=0.5, help="Initial temperature: T_0")
    parser.add_argument('--final_temp', type=float, default=1.5, help="Final temperature: T_f")
    parser.add_argument('--temp_step', type=float, default=0.01, help="Temperature step: dT")
    parser.add_argument('--h', type=float, default=0.0, help="externel field strength")
    parser.add_argument('--steps', type=int, default=2000, help="Number of glauber dynamics steps")
    # parser.add_argument('--dt', type=float, default=0.01, help="Time step of glauber dynamics")
    parser.add_argument('--save_interval', type=int, default=10, help="Interval of saving the microscopic configuration")
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--n_proc', type=int, default=100, help="Number of processors for multiprocessing")
    parser.add_argument('--mag', type=int, default=None, help="Target magnetization")
    parser.add_argument('--flag_test', action='store_true', help="A flag for saving the test data")
    parser.add_argument('--save_config', default=False, help="A flag for saving the configuration data")
    parser.add_argument('--n_events', type=int, default=None, help="Number of events for Glauber dynamics")
    args = parser.parse_args()
    return args

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# Monte Carlo task
def mcmc_task(args, Ts, seed_list):

    Es, Ms, Cs, Xs, Ms_time = [], [], [], [], []

    m = Glauber2DCW(args)
    pbar = tqdm(desc="Progress: ".format(id), total=len(Ts))
    for T, seed in zip(Ts, seed_list):
        # Set seeds for each individual run to ensure reproducibility
        np.random.seed(int(seed))
        random.seed(int(seed))

        E, M, C, X, M_time, _, _ = m.simulate(1 / T, h=args.h, n_events=args.n_events)
        Es.append(E)
        Ms.append(abs(M))
        Cs.append(C)
        Xs.append(X)
        Ms_time.append(M_time)
        pbar.update(1)

    return Es, Ms, Cs, Xs, Ms_time


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

    T_0 = args.init_temp
    T_f = args.final_temp
    dT = args.temp_step
    NT = int((T_f - T_0) / dT) + 1
    Ts_base = [T_0 + dT * step for step in range(NT)]
    Ts = Ts_base * args.repeat  # Repeat each temperature args.repeat times
    Ts_list = np.array_split(np.array(Ts), n_proc)

    seeds = [i for i in range(len(Ts))]
    seeds_list = np.array_split(np.array(seeds), n_proc)

    with poolcontext(processes=n_proc) as pool:
        Es_list, Ms_list, Cs_list, Xs_list, Ms_time_list = zip(*pool.starmap(mcmc_task, zip(args_list, Ts_list, seeds_list)))
    Es, Ms, Cs, Xs, Ms_time, = sum(Es_list, []), sum(Ms_list, []), sum(Cs_list, []), sum(Xs_list, []), sum(Ms_time_list, [])
    print("\n> Elapsed time: {:4f}s".format(time.time() - start))
    
    from collections import defaultdict
    temp_data = defaultdict(lambda: {'E': [], 'M': [], 'C': [], 'X': [], 'U': []})

    # Group data by temperature
    for t, e, m, c, x in zip(Ts, Es, Ms, Cs, Xs):
        temp_data[t]['E'].append(e)
        temp_data[t]['M'].append(m)
        temp_data[t]['C'].append(c)
        temp_data[t]['X'].append(x)

    # Calculate means for each temperature
    Ts_unique = sorted(temp_data.keys())
    Es_mean = [np.mean(temp_data[t]['E']) for t in Ts_unique]
    Ms_mean = [np.mean(temp_data[t]['M']) for t in Ts_unique]
    Cs_mean = [np.mean(temp_data[t]['C']) for t in Ts_unique]
    Xs_mean = [np.mean(temp_data[t]['X']) for t in Ts_unique]

    # Also calculate standard deviations for error analysis
    Es_std = [np.std(temp_data[t]['E']) for t in Ts_unique]
    Ms_std = [np.std(temp_data[t]['M']) for t in Ts_unique]
    Cs_std = [np.std(temp_data[t]['C']) for t in Ts_unique]
    Xs_std = [np.std(temp_data[t]['X']) for t in Ts_unique]

    # Record and plot the result
    rootpath = './result'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    # Save the result into csv file
    csv_name = rootpath + "/repeat_{}_result_L{}_h{}.csv".format(args.repeat, args.L, args.h)
    f = open(csv_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['T', 'E', 'M', 'C', 'X'])
    for t, e, m, c, x in zip(Ts, Es, Ms, Cs, Xs):
        writer.writerow([t, e, m, c, x])
    f.close()
    print("Saved the result into {}.".format(csv_name))

    # Save the result into a plot
    fig = plt.figure(figsize=(18, 10))
    sp = fig.add_subplot(2, 2, 1)
    plt.errorbar(Ts_unique, Es_mean, fmt='o', color='IndianRed', capsize=5)
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy", fontsize=20)
    plt.axis('tight')

    sp = fig.add_subplot(2, 2, 2)
    plt.errorbar(Ts_unique, Ms_mean, fmt='o', color='RoyalBlue', capsize=5)
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization", fontsize=20)
    plt.axis('tight')

    sp = fig.add_subplot(2, 2, 3)
    plt.errorbar(Ts_unique, Cs_mean, fmt='o', color='IndianRed', capsize=5)
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat", fontsize=20)
    plt.axis('tight')

    sp = fig.add_subplot(2, 2, 4)
    # plt.errorbar(Ts_unique, Xs_mean, yerr=Xs_std, fmt='o', color='RoyalBlue', capsize=5)
    plt.errorbar(Ts_unique, Xs_mean, fmt='o', color='RoyalBlue', capsize=5)
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.axis('tight')

    plot_name = rootpath + "/repeat_{}_plot_L{}_h{}.png".format(args.repeat, args.L, args.h)
    plt.savefig(plot_name)
    plt.clf()
    print("Saved the plot into {}.".format(plot_name))

