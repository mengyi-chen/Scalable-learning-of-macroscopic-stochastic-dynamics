# Acknowledgment: 
# This code is based on the repository available at https://github.com/MLDS-NUS/Learn-Partial/blob/master/Predator_Prey/raw_data/batch_solver.py.

import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import os, sys
sys.path.append('..')
import torch
from utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import random
import yaml

# Load parameters from YAML configuration file
with open('../config/config.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Set up argument Parser for command-line options
parser = argparse.ArgumentParser(description='Spatial Predator Prey model')
parser.add_argument('--maxit', default=params['maxit'], type=int, help='Number of iterations')
parser.add_argument('--seed', default=params['seed'], type=int, help='Random seed')
parser.add_argument('--nx', default=params['nx'], type=int, help='Number of grid')
parser.add_argument('--bs', default=params['bs'], type=int, help='Number of trajectories')
parser.add_argument('--a', default=params['a'], type=float, help='parameter a')
parser.add_argument('--b', default=params['b'], type=float, help='parameter b')
parser.add_argument('--m', default=None, type=float, help='parameter m for initial condition')
parser.add_argument('--k', default=None, type=float, help='parameter k for initial condition')
parser.add_argument('--noise_level', default=params['noise_level'], type=float, help='noise level')
parser.add_argument('--D', default=params['D'], type=float, help='parameter D')
parser.add_argument('--dt', default=params['dt'], type=float, help='dt')
parser.add_argument('--flag_test', action='store_true', help='test')
parser.add_argument('--solver', default=params['solver'], type=str, choices=['euler', 'RK4'])
parser.add_argument('--gpu_idx', default=5, type=int, help='GPU index')
args = parser.parse_args()
print(args)

def initial_value(Nx, x, bs=args.bs):
    """Initial conditions for the predator-prey model.

    Args:
        Nx (int): Number of spatial grid points.
        x (np.ndarray): Spatial grid points.
        bs (int): Number of trajectories. Defaults to args.bs.

    Returns:
        uv_init: initial conditions array of shape [bs, 2, Nx]
    """

    uv_init = np.zeros([bs, 2, Nx])
    if args.flag_test:
        # Use fixed parameters for testing
        assert args.m is not None
        assert args.k is not None
        k_list = args.k * np.ones(bs)
        m_list = args.m * np.ones(bs) 
        
    else:
        # Randomly generate parameters for training
        k_list = np.random.uniform(0.05, 0.15, bs)
        m_list = np.random.uniform(0.45, 0.55, bs)

    for i in range(k_list.shape[0]):
        k = k_list[i]
        m = m_list[i]

        # Initialize predator and prey populations
        uv_init[i, 0] = m + k * np.cos(x * np.pi * 10) 
        uv_init[i, 1] = (1 - m) - k * np.cos(x * np.pi * 10) 

    return uv_init  

def predator_prey_solve(dt, maxit, uv_init):
    """Solve the predator-prey model using specified numerical methods.

    Args:
        dt (float): Time step size.
        maxit (int): Number of iterations.
        uv_init (np.ndarray): Initial conditions array of shape [bs, 2, Nx].
        
    Returns:
        X0: Tensor of shape [bs, maxit, 2, Nx] containing the states at each time step.
        X1: Tensor of shape [bs, maxit, 2, Nx] containing the states after one additional time step.

    """

    # Set device to GPU if available, otherwise use CPU
    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    model = Net(dt, device, a=args.a, b=args.b, D=args.D).to(device)
    
    # Initial value
    # uv_init: # [B, 2, Nx]
    img = torch.FloatTensor(uv_init).to(device) # [B, 2, Nx]

    start = time.time()
    u = []

    with torch.no_grad():

        for step in tqdm(range(maxit)): # Iterate for maxit steps

            if args.solver == 'euler':
                # Use Euler method for solving
                u0, u1 = euler(model, img, dt, noise_level=args.noise_level)

            elif args.solver == 'RK4':
                # Use Runge-Kutta 4th order method for solving
                u0, u1 = rk4(model, img, dt, noise_level=args.noise_level)

            img = u1
            u.append(u0)
            

    runtime = time.time() - start
    print("Pytorch Runtime: ", runtime)

    u = torch.stack(u)  # [maxit, B, 2, Nx]
    X0 = torch.transpose(u, 1, 0)  # [B, maxit, 2, Nx]
    X1 = []
    for i in tqdm(range(X0.shape[0])):
        u0 = X0[i].clone() # [len_per_tra, 1, nx]
        if args.solver == 'euler':
            _, u1 = euler(model, u0, args.dt, noise_level=args.noise_level)
        elif args.solver == 'RK4':
            _, u1 = rk4(model, u0, args.dt, noise_level=args.noise_level)
        X1.append(u1)
    X1 = torch.stack(X1)

    return X0, X1


if __name__ == "__main__":
    
    set_seed(args.seed)
    dx = 1 / args.nx 
    x = np.linspace(-0.5 * dx, dx * (args.nx + 0.5), args.nx + 2)[1:-1] # Nx 
    dt = args.dt
    maxtime = dt * args.maxit

    # Initial conditions
    uv_init = initial_value(args.nx, x) # [B, 2, Nx]
    
    # Solve the predator-prey model
    X0, X1 = predator_prey_solve(dt, args.maxit, uv_init) # [n_tra, len_per_tra, 2, Nx]
    print('X0 shape:', X0.shape)
    print('X1 shape:', X1.shape)

    if not args.flag_test:
        if args.nx == 100:
            torch.save(X0, f'X0_grid_{args.nx}.pt')
        else:
            torch.save(X0, f'X0_val_grid_{args.nx}.pt')
            torch.save(X1, f'X1_val_grid_{args.nx}.pt')
    else:
        # Save results for testing
        folder = './test_data'
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(X0, os.path.join(folder, f'X0_test_grid_{args.nx}_m_{args.m}_k_{args.k}.pt'))

    # Visualize results
    X0 = X0.detach().cpu().numpy()
    uv_mean = np.mean(X0, -1) # [len, n_tra, 2]
    fig = plt.figure(figsize=(24, 6))
    axes = fig.add_subplot(1, 3, 1)
    for i in range(20):
        axes.plot(uv_mean[i, :, 0])

    axes = fig.add_subplot(1, 3, 2)
    for i in range(20):
        axes.plot(uv_mean[i, :, 1])

    axes = fig.add_subplot(1, 3, 3)
    for i in range(20):
        axes.plot(uv_mean[i, :, 0], uv_mean[i, :, 1])

    plt.savefig('uv_mean_stochastic.png')



