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

from utils.utils import *
from utils.predict_energy_barrier import MLPModel, predict_barriers  # Numba version
from utils.kmc_cpu import KMC_CPU

# Choose your backend here:
parser = argparse.ArgumentParser(description='CPU-based KMC Simulation')
parser.add_argument('--box_L', default=8, type=int)
parser.add_argument('--number_of_steps', default=2000000, type=int)
parser.add_argument('--temperature', default=2000, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--output_dir', default='output', type=str)
parser.add_argument('--num_vacancies', default=None, type=int)
parser.add_argument('--vacancy_percentage', default=1/1024, type=float)
parser.add_argument('--number_of_log_steps', default=50, type=int)
parser.add_argument('--number_of_dump_steps', default=50, type=int)
parser.add_argument('--dump_flag', default=True, help='Whether to dump the configuration')
args = parser.parse_args()

def main():
    set_seed(args.seed)


    attempt_frequency = 1e13
    boltzmann_constant = 8.617333e-5
    N_atoms = 2 * args.box_L ** 3  

    # read simulation settings
    input = f'../input/input_{N_atoms}/INPUT_{N_atoms}'
    settings = dict(read_settings(input))

    # output directory
    initial_configuration = f'../input/input_{N_atoms}/initial_N_{N_atoms}_equimolar.dump'

    save_dir = os.path.join(args.output_dir, "config_data")
    os.makedirs(save_dir, exist_ok=True)
   
    settings.update({
        'number_of_steps': args.number_of_steps,
        'temperature'    : args.temperature,
        'output_dir'     : args.output_dir,
        "initial_configuration" : initial_configuration,
        'log_file'       : os.path.join(args.output_dir, 'log.csv'),
        'chemical_order_file': os.path.join(args.output_dir, 'chemical_order.csv'),
        'number_of_log_steps': args.number_of_log_steps,
        'number_of_dump_steps': args.number_of_dump_steps
    })
    settings = begin_of_program(settings)

    voxel_size = float(settings["voxel_size"])
    cutoff = float(settings["image_cutoff"])
    number_of_voxels_in_one_side = np.round(cutoff / voxel_size - 0.5).astype(np.int32)
    occupied_masks = calculate_masks(cutoff, voxel_size, number_of_voxels_in_one_side * 2 + 1)

    # Pre-compute static data for performance
    crop_radius = number_of_voxels_in_one_side

    indexes = (
        settings["dimensions_start_line"],
        settings["dimensions_end_line"], 
        settings["coordinates_start_line"],
        settings["coordinates_end_line"]
    )
    dimensions, configurations = read_configurations(settings["initial_configuration"], indexes)

    model_weights = np.load(settings["model_weights"], allow_pickle=True)
    model = MLPModel(model_weights)
    if args.num_vacancies is not None:
        num_vacancies = args.num_vacancies
    else:
        num_vacancies = int(N_atoms * args.vacancy_percentage)
    print('#' * 20)
    print(f"Number of vacancies: {num_vacancies}")
    print('#' * 20)


    # save initial chemical order
    config_grid = map_coords_to_grid(configurations, ncell=args.box_L)
    vacancy_per_axis = int(np.round(num_vacancies ** (1/3)))

    vacancy_x = np.arange(0, vacancy_per_axis, dtype=np.int32) * int(config_grid.shape[0] / vacancy_per_axis)
    vacancy_x, vacancy_y, vacancy_z = np.meshgrid(vacancy_x, vacancy_x, vacancy_x, indexing='ij')
    vacancy_indices = np.stack((vacancy_x.flatten(), vacancy_y.flatten(), vacancy_z.flatten()), axis=1).astype(np.int32)
    config_grid[vacancy_indices[:, 0], vacancy_indices[:, 1], vacancy_indices[:, 2]] = 0  # Set vacancies to 0

    if args.dump_flag == True:
        np.save(os.path.join(save_dir, f"output_0.npy"), config_grid)
        
    # save initial dump file
    initial_configurations = map_grid_to_coords(config_grid)
    dump(os.path.join(args.output_dir, "output_0.dump"), 0, settings["number_of_atoms"], dimensions, initial_configurations)
    local_order = cal_local_chemical_order(config_grid)
    write_chemical_order(settings["chemical_order_file"], [0] + list(local_order))

    ################################################################################
    #                             Begin KMC Simulation                             #
    ################################################################################
    kmc_time = 0
    start_time = time.time()
    KMC_process = KMC_CPU(attempt_frequency, boltzmann_constant, settings["temperature"], settings["time_scale"])

    # -1: unoccupied, 0: vacancy, 1: atom type 1, 2: atom type 2, ..., n: atom type n
    for step in tqdm(range(settings["initial_step"], settings["number_of_steps"] + settings["initial_step"])):

        # find the neighboring atoms of the vacancy within the cutoff distance
        cropped_data = get_cropped_pbc_region(config_grid, vacancy_indices, crop_radius, occupied_masks) # [N, 9, 9, 9]
        image_array, directions, vacancy_mask = rotate_and_mirror_data(cropped_data, number_of_voxels_in_one_side) # [N, 8, 729], [8,3]
   
        # predict diffusion barriers
        energy_barriers = predict_barriers(model, image_array) # [N, 8]
        # Set energy barriers to infinity where the target position is already a vacancy
        energy_barriers[vacancy_mask] = float('inf')
        pair_information = [directions, energy_barriers]

        # kinetic Monte Carlo step 
        jump_info, jumping_time, config_grid, vacancy_indices = KMC_process.execute(
            config_grid, pair_information, vacancy_indices
        )
        
        kmc_time += jumping_time
        old_pos, new_pos, neighbor_type = jump_info
        
        # log the step information
        variables = [step, kmc_time, 0] + (old_pos * voxel_size).tolist()
        variables.append(int(neighbor_type))
        variables.extend((new_pos * voxel_size).tolist())

        # write log file every number_of_log_steps
        if step % settings["number_of_log_steps"] == 0:
            write_csv(settings["log_file"], variables) 
            local_order = cal_local_chemical_order(config_grid)
            write_chemical_order(settings["chemical_order_file"], [step] + list(local_order))
            
        # save configuration every number_of_dump_steps
        if args.dump_flag == True and step % settings["number_of_dump_steps"] == 0:
            np.save(os.path.join(save_dir, f"output_{step}.npy"), config_grid)

    # save final configuration
    final_configurations = map_grid_to_coords(config_grid)
    dump(os.path.join(args.output_dir, f"output_{step}.dump"), step, settings["number_of_atoms"], dimensions, final_configurations)

    print(f"Simulation completed in {kmc_time:.2f} time units.")
    print(f"Total wall time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
