from marshal import dump
import os
import sys
sys.path.append('../')
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from utils.utils import *
from typing import Tuple, Optional
from utils.predict_energy_barrier import MLPModel, predict_barriers
from utils.kmc_cpu import KMC_CPU

# Choose your backend here:
parser = argparse.ArgumentParser(description='CPU-based KMC Simulation')
parser.add_argument('--L', default=32, type=int)
parser.add_argument('--max_steps', default=125, type=int)
parser.add_argument('--min_steps', default=5, type=int)
parser.add_argument('--model_weights', default='../utils/weights.npy', type=str)
parser.add_argument('--unit', default='s', type=str)
parser.add_argument('--patch_L', default=8, type=int)
parser.add_argument('--equilibration_steps', default=1000, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--N_samples', default=2000, type=int)
parser.add_argument('--n_events', default=8, type=int)
parser.add_argument('--temperature', default=2000, type=int)
args = parser.parse_args()

def redistribute_atoms(config_grid):
    # redistribute the atom types 
    N_atoms = np.argwhere(config_grid >= 0).shape[0]

    idx_0 = np.argwhere(config_grid == 0) # [n, 3]
    idx_1 = np.argwhere(config_grid == 1)
    idx_2 = np.argwhere(config_grid == 2)
    idx_3 = np.argwhere(config_grid == 3)

    N_target = (N_atoms - idx_0.shape[0]) // 3

    if idx_1.shape[0] < N_target:
        idx = np.random.choice(idx_2.shape[0], size=N_target - idx_1.shape[0], replace=False)
        indices = idx_2[idx]
        config_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    elif idx_1.shape[0] > N_target:
        idx = np.random.choice(idx_1.shape[0], size=idx_1.shape[0] - N_target, replace=False)
        indices = idx_1[idx]
        config_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 2

    idx_2 = np.argwhere(config_grid == 2) # [n, 3]
    if idx_2.shape[0] < N_target:
        idx = np.random.choice(idx_3.shape[0], size=N_target - idx_2.shape[0], replace=False)
        indices = idx_3[idx]
        config_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 2
    elif idx_2.shape[0] > N_target:
        idx = np.random.choice(idx_2.shape[0], size=idx_2.shape[0] - N_target, replace=False)
        indices = idx_2[idx]
        config_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 3

    return config_grid


def redistribution_vacancies(config_grid):
    # redistribute the atom types 

    idx_0 = np.argwhere(config_grid == 0) # [n, 3]
    # target_places = [(8, 8, 8), (8, 8, 24), (8, 24, 8), (8, 24, 24),
    #                  (24, 8, 8), (24, 8, 24), (24, 24, 8), (24, 24, 24)]

    starts = range(args.patch_L, config_grid.shape[0], args.patch_L*2)
    target_places = [(i, j, k) for i in starts for j in starts for k in starts]

    for i in range(len(target_places)):
        assert config_grid[target_places[i]] >= 0
        config_grid[idx_0[i][0], idx_0[i][1], idx_0[i][2]] = config_grid[target_places[i]]
        config_grid[target_places[i]] = 0
    return config_grid

def generate_pool(T):
    pool = []
    if T <= 1200 and T >= 300:    
        folder = '../data/output_atoms_1024_steps_20000000'
    elif T >= 1400 and T <= 3000:
        folder = '../data/output_atoms_1024_steps_2000000'
    else:
        raise ValueError("Temperature out of range. Please choose a temperature between 300 and 3000 K.")
    for seed in range(10):
        micro_val_path = os.path.join(folder, f'T_{T}_seed_{seed}', 'micro_val.npy')
        micro_val = np.load(micro_val_path, allow_pickle=True)
        pool.append(micro_val)
            
    pool = np.stack(pool) # [10, 2001, 16, 16, 16]

    return pool


def LocalRelax(spins, T, patch_size, stride, n_events=10):
    # spins shape: [L * 2, L * 2, L * 2]
    L = spins.shape[0]
    starts = range(stride, L, patch_size)
  
    for i in starts:
        for j in starts:
            for k in starts:
                x_indices = np.arange(i, i + patch_size) % L
                y_indices = np.arange(j, j + patch_size) % L
                z_indices = np.arange(k, k + patch_size) % L

                patch_spins = spins[np.ix_(x_indices, y_indices, z_indices)]
                idx_vacancy = np.where(patch_spins == 0)[0]
                if idx_vacancy.size == 0:
                    continue
                # Process all patches in the batch at once
                patch_spins, _, _ = kmc_simulation(
                    patch_spins, T, n_events, num_voxels=4
                )

                spins[np.ix_(x_indices, y_indices, z_indices)] = patch_spins
    return spins


def UpSample(L, pool):

    patch_L = pool.shape[-1] // 2
    assert L % patch_L == 0, "L must be a multiple of patch_L"
    tile_times = L // patch_L
        
    length = pool.shape[1]
    idx = np.random.randint(length)

    idx_min = np.max([0, idx - 10])
    idx_max = np.min([length, idx + 10])
    choices = pool[:, idx_min:idx_max].reshape(-1, 2*patch_L, 2*patch_L, 2*patch_L)  # [N, 16, 16, 16]

    config_grid_large = np.empty((L*2,L*2,L*2), dtype=np.int32)
    
    for _, (i,j,k) in enumerate(product(range(tile_times), repeat=3)):
        # Randomly choose one from the closest configurations
        indice = np.random.choice(choices.shape[0], size=1, replace=True)[0] 
        s = choices[indice]
                
        # random cyclic shift
        dx, dy, dz = np.random.randint(0, patch_L, size=3) * 2
        s = np.roll(np.roll(np.roll(s, dx, 0), dy, 1), dz, 2)

        index_x = np.arange(i*patch_L*2,(i+1)*patch_L*2)
        index_y = np.arange(j*patch_L*2,(j+1)*patch_L*2)
        index_z = np.arange(k*patch_L*2,(k+1)*patch_L*2)

        # Use np.ix_ to create proper 3D indexing for block assignment
        config_grid_large[np.ix_(index_x, index_y, index_z)] = s

    config_grid_large = redistribute_atoms(config_grid_large)
 
    if args.n_events > 0:
        config_grid_large = LocalRelax(config_grid_large, args.temperature, patch_size=args.patch_L*2, stride=args.patch_L, n_events=args.n_events)
    config_grid_large = redistribution_vacancies(config_grid_large)

    return config_grid_large


class KMC_CPU_boundary():

    def __init__(self, temperature):

        self.attempt_frequency = 1e13
        self.boltzmann_constant = 8.617333e-5      
        scale = {"Ms" : 1e-6, "Ks" : 1e-3, "s" : 1e0, "ms" : 1e3, "us" : 1e6, "ns" : 1e9, "ps" : 1e12, "fs" : 1e15}
        self.temperature = temperature
        self.time_scale = scale[args.unit]  

    def execute(self, config_grid, pair_information, vacancy_indices, k=8):
        # config_grid: [2L, 2L, 2L] numpy array
        # vacancy_indices: [N, 3]
        directions, barriers = pair_information  # directions: (D, 3), barriers: (N, 8)
        break_boundary = False
        
        # Compute jump rates
        rates = self.attempt_frequency * np.exp(-barriers / (self.boltzmann_constant * self.temperature)) # [N, 8]
        rates_flat = rates.flatten()  # [N * 8]
        total_rate = np.sum(rates_flat)

        probs = rates_flat / total_rate

        # Sample one jump
        rand = np.random.rand(1)[0]
        cum_probs = np.cumsum(probs)
        selected_flat_idx = np.searchsorted(cum_probs, rand)
        selected_flat_idx = min(selected_flat_idx, len(cum_probs) - 1)  # Ensure index is within bounds

        vacancy_idx = selected_flat_idx // 8
        direction_idx = selected_flat_idx % 8

        
        selected_direction = directions[direction_idx]

        # Compute time increment
        # random_number = np.random.rand(1)[0]
        # jumping_time = - np.log(random_number) / total_rate * self.time_scale

        # NOTE: use the mean waiting time 
        jumping_time = 1 / total_rate * self.time_scale

        # Update grid: swap vacancy and neighbor
        old_pos = vacancy_indices[vacancy_idx].copy()  # Make a copy to avoid confusion
        new_pos = old_pos + selected_direction

        # if np.any(new_pos < 0) or np.any(new_pos >= np.array(config_grid.shape)):
        #     break_boundary = True

        new_pos = new_pos % config_grid.shape[0]  # Apply periodic boundary conditions

        neighbor_type = config_grid[new_pos[0], new_pos[1], new_pos[2]]
        config_grid[old_pos[0], old_pos[1], old_pos[2]] = neighbor_type
        config_grid[new_pos[0], new_pos[1], new_pos[2]] = 0
    
        # Update vacancy_indices
        vacancy_indices[vacancy_idx] = new_pos

        return (old_pos, new_pos, neighbor_type), jumping_time, config_grid, vacancy_indices, break_boundary



def partial_kmc_simulation(initial_config_grid, temperature, num_voxels=4):

    # # sample a partial vacancy
    vacancy_indices = np.argwhere(initial_config_grid == 0).reshape(-1, 3)  # Reshape to ensure it is a 2D array with one vacancy
    partial_vacancy_idx = np.random.choice(vacancy_indices.shape[0])
    center = vacancy_indices[partial_vacancy_idx, :]

    # generate partial config grid
    ncell = initial_config_grid.shape[0]

    idx_range = (np.arange(center[0] - args.patch_L, center[0] + args.patch_L) % ncell)
    idy_range = (np.arange(center[1] - args.patch_L, center[1] + args.patch_L) % ncell)
    idz_range = (np.arange(center[2] - args.patch_L, center[2] + args.patch_L) % ncell)
    config_grid = initial_config_grid[np.ix_(idx_range, idy_range, idz_range)]

    # NOTE: use the mean 
    # z0 = np.array(cal_local_chemical_order(initial_config_grid))
    z0 = []
    for i, j, l in product(range(args.L // args.patch_L), repeat=3):
        sub_idx = (np.arange(i*args.patch_L*2, (i+1)*args.patch_L*2))
        sub_idy = (np.arange(j*args.patch_L*2, (j+1)*args.patch_L*2))
        sub_idz = (np.arange(l*args.patch_L*2, (l+1)*args.patch_L*2))
        sub_grid = initial_config_grid[np.ix_(sub_idx, sub_idy, sub_idz)]
        z0.append(cal_local_chemical_order(sub_grid))
    z0 = np.mean(z0, axis=0)  # Calculate the mean z


    z0_partial = np.array(cal_local_chemical_order(config_grid))  # Calculate z0 for the partial config grid
    config_grid, step, kmc_time = kmc_simulation(config_grid, temperature, args.max_steps, num_voxels=num_voxels)

    z1_partial = np.array(cal_local_chemical_order(config_grid))
    z1_hat = z0 + (z1_partial - z0_partial)

    return step, kmc_time, z0, z1_hat


def kmc_simulation(config_grid, temperature, steps, num_voxels=4):
    
    ################################################################################
    #                             Begin KMC Simulation                             #
    ################################################################################

    kmc_time = 0    
    occupied_masks = calculate_masks()
    model_weights = np.load(args.model_weights, allow_pickle=True)
    model = MLPModel(model_weights)

    KMC_process = KMC_CPU_boundary(temperature)
    partial_vacancy = np.argwhere(config_grid == 0).reshape(-1, 3)  # Reshape to ensure it is a 2D array with one vacancy   
    # print('partial_vacancy shape:', partial_vacancy.shape) 
    
    # -1: unoccupied, 0: vacancy, 1: atom type 1, 2: atom type 2, ..., n: atom type n
    for step in range(1, steps):

        # find the neighboring atoms of the vacancy within the cutoff distance
        cropped_data = get_cropped_pbc_region(config_grid, partial_vacancy, num_voxels, occupied_masks) # [N, 9, 9, 9]
        image_array, directions, vacancy_mask = rotate_and_mirror_data(cropped_data, num_voxels) # [N, 8, 729], [8,3]
   
        # predict diffusion barriers
        energy_barriers = predict_barriers(model, image_array) # [N, 8]

        # Set energy barriers to infinity where the target position is already a vacancy
        energy_barriers[vacancy_mask] = float('inf')
        pair_information = [directions, energy_barriers]

        jump_info, jumping_time, config_grid, partial_vacancy, break_boundary = KMC_process.execute(
            config_grid, pair_information, partial_vacancy
        )

        old_pos, new_pos, neighbor_type = jump_info
        kmc_time += jumping_time   

        # if break_boundary == True:
        #     break
    
    return config_grid, step, kmc_time


def main():
    set_seed(args.seed)
    # ========= Load the configuration of small system =========
    pool = generate_pool(args.temperature)  # [10, 2001, 16, 16, 16], [10, 2001, n_features]

    # ========= Generate the configuration of large system =========
    steps = []
    kmc_times = []
    macro_vals_partial = []
    macro_vals = []
    
   
    pbar = tqdm(total=args.N_samples, desc="Processing configurations")
    while len(macro_vals) < args.N_samples:

        x0 = UpSample(args.L, pool)
        step, kmc_time, z0, z1_partial = partial_kmc_simulation(x0, args.temperature)

        if step < args.min_steps:
            continue

        steps.append(step)
        kmc_times.append(kmc_time)
        macro_vals_partial.append(z1_partial)
        macro_vals.append(z0)    
        pbar.update(1)
    pbar.close()
    
    steps = np.array(steps)
    kmc_times = np.array(kmc_times)
    macro_vals_partial = np.array(macro_vals_partial)
    macro_vals = np.array(macro_vals)

    folder = f'../data/partial_sampling_atoms_{2*args.L**3}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, f'step_T_{args.temperature}_seed_{args.seed}.npy'), steps)
    np.save(os.path.join(folder, f'kmc_times_T_{args.temperature}_seed_{args.seed}.npy'), kmc_times)
    np.save(os.path.join(folder, f'z1_train_partial_T_{args.temperature}_seed_{args.seed}.npy'), macro_vals_partial)
    np.save(os.path.join(folder, f'z0_train_T_{args.temperature}_seed_{args.seed}.npy'), macro_vals)

if __name__ == "__main__":
    main()
