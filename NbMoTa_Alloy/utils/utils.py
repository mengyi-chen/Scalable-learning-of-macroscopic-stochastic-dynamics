import numpy as np
import random
import torch
import os
import torch.nn as nn
import torch.optim as optim

def set_seed(seed):
    """Set the seed for reproducibility across NumPy, Python, and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def begin_of_program(settings):
    """Initialize program settings and prepare output files.
    
    Args:
        settings (dict): Configuration settings dictionary
    
    Returns:
        dict: Processed settings with converted types
    """
    # Integer parameter list
    int_parameter_list = [
        "number_of_atoms",
        "dimensions_start_line",
        "dimensions_end_line",
        "coordinates_start_line",
        "coordinates_end_line",
        "initial_step",
        "number_of_steps",
        "number_of_log_steps",
        "number_of_dump_steps",
        "number_of_steps_updating_vacancy"
    ]

    float_parameter_list = [
        "image_cutoff",
        "first_nearest_neighbor_cutoff",
        "voxel_size"
    ]

    string_parameter_list = [
        "initial_configuration",
        "model_weights",
        "units",
        "log_file",
        "chemical_order_file"
    ]

    # List all required parameters
    name_list = int_parameter_list + float_parameter_list + string_parameter_list

    for name in name_list:
        print("%-40s" % name, settings[name])

    # Convert type of parameters
    for name in int_parameter_list:
        settings[name] = int(settings[name])
    for name in float_parameter_list:
        settings[name] = float(settings[name])
    
    # Time scale conversion
    scale = {
        "Ms": 1e-6, "Ks": 1e-3, "s": 1e0, "ms": 1e3, 
        "us": 1e6, "ns": 1e9, "ps": 1e12, "fs": 1e15
    }
    settings["time_scale"] = scale[settings["units"]]

    # Open output files
    with open(settings["log_file"], "w", newline='') as f:
        header = ("step,time,jumping_vacancy_type,vacancy_x,vacancy_y,vacancy_z,"
                 "exchange_neighbor_type,neighbor_x,neighbor_y,neighbor_z\n")
        f.write(header)
    
    with open(settings["chemical_order_file"], "w", newline='') as f:
        header = ("step,delta_NbNb,delta_NbMo,delta_NbTa,"
                 "delta_MoMo,delta_MoTa,delta_TaTa\n")
        f.write(header)

    return settings


def read_settings(file_name):
    """Read settings from a configuration file.
    
    Args:
        file_name (str): Path to the settings file
    
    Returns:
        list: List of parsed settings lines
    """
    with open(file_name) as f:
        lines = f.readlines()
    
    settings = [line.strip().split() for line in lines]
    return settings

def read_configurations(file_name, indexes):
    """Read atomic configurations from a LAMMPS-style configuration file.
    
    Args:
        file_name (str): Path to the configuration file
        indexes (list): List of line indices [dim_start, dim_end, config_start, config_end]
    
    Returns:
        tuple: (dimensions, configurations) as numpy arrays
    """
    with open(file_name) as f:
        lines = f.readlines()
     
    dimensions = [line.strip().split() for line in lines[indexes[0]-1 : indexes[1]]]
    configurations = [line.strip().split() for line in lines[indexes[2]-1 : indexes[3]]]

    dimensions, configurations = np.array(dimensions).astype(np.float32), np.array(configurations).astype(np.float32)
     
    return dimensions, configurations


def dump(output_file_name, time, number_of_atoms, dimensions, configurations):
    """Dump atomic configuration to a LAMMPS-style output file.
    
    Args:
        output_file_name (str): Output file path
        time (int): Current timestep
        number_of_atoms (int): Total number of atoms
        dimensions (np.ndarray): Box dimensions
        configurations (np.ndarray): Atomic configurations
    """
    with open(output_file_name, 'w') as output_file:
        output_file.write("ITEM: TIMESTEP\n")
        output_file.write(str(time) + "\n")
        output_file.write("ITEM: NUMBER OF ATOMS\n")
        output_file.write(str(number_of_atoms) + "\n")
        output_file.write("ITEM: BOX BOUNDS pp pp pp\n")
        output_file.write(f"{dimensions[0, 0]} {dimensions[0, 1]}\n")
        output_file.write(f"{dimensions[1, 0]} {dimensions[1, 1]}\n")
        output_file.write(f"{dimensions[2, 0]} {dimensions[2, 1]}\n")
        output_file.write("ITEM: ATOMS id type x y z\n")
        
        for row in range(len(configurations)):
            for column in range(4):
                output_file.write(str(configurations[row, column]) + " ")
            output_file.write(str(configurations[row, -1]) + "\n")


def write_csv(log_file, parameters):
    """Write parameters to CSV log file with proper formatting.
    
    Args:
        log_file (str): Path to the log file
        parameters (list): List of parameters to write
    """
    with open(log_file, "a") as f:
        formatted_params = []
        for idx, p in enumerate(parameters):
            if idx in [2, 6]:  # Integer formatting
                formatted_params.append(str(int(p)))
            elif idx in [3, 4, 5, 7, 8, 9]:  # Float with 2 decimals
                formatted_params.append("{:.2f}".format(p))
            elif idx in [1]:  # Scientific notation
                formatted_params.append("{:.3e}".format(p))
            else:
                formatted_params.append(str(p))
        
        log_line = ",".join(formatted_params) + "\n"
        f.write(log_line)


def write_chemical_order(log_file, parameters):
    """Write chemical order parameters to CSV file.
    
    Args:
        log_file (str): Path to the log file
        parameters (list): List of chemical order parameters
    """
    with open(log_file, "a") as f:
        formatted_params = []
        for p in parameters:
            if isinstance(p, float):
                formatted_params.append("{:.3e}".format(p))
            else:
                formatted_params.append(str(p))

        log_line = ",".join(formatted_params) + "\n"
        f.write(log_line)

def map_coords_to_grid(configuration, ncell=8, a=3.24, voxel_size=1.62):
    """Map atomic coordinates to a 3D grid.
    
    Args:
        configuration (np.ndarray): Atomic configuration array
        ncell (int): Number of unit cells
        a (float): Lattice parameter
        voxel_size (float): Size of each voxel
    
    Returns:
        np.ndarray: 3D grid with atom types
    """
    crystal_grid = np.full((ncell * 2, ncell * 2, ncell * 2), -1, dtype=np.int32)
    coords = configuration[:, 2:5]  # [N, 3]
    atom_types = configuration[:, 1].astype(np.int32)
    ijk_indices = np.round(coords / voxel_size).astype(np.int32)

    i, j, k = ijk_indices[:, 0], ijk_indices[:, 1], ijk_indices[:, 2]
    crystal_grid[i, j, k] = atom_types

    return crystal_grid


def map_grid_to_coords(crystal_grid, voxel_size=1.62):
    """Convert 3D grid back to atomic coordinates.
    
    Args:
        crystal_grid (np.ndarray): 3D grid with atom types
        voxel_size (float): Size of each voxel
    
    Returns:
        np.ndarray: Atomic configuration array
    """
    # Get indices where the grid is occupied (not -1)
    occupied_indices = np.argwhere(crystal_grid != -1)
    
    # Get corresponding atom types
    atom_types = crystal_grid[occupied_indices[:, 0],
                              occupied_indices[:, 1],
                              occupied_indices[:, 2]]
    
    # Convert grid indices back to real positions (center of voxel)
    positions = occupied_indices * voxel_size
    
    # Stack atom types and positions into configuration array
    configuration = np.column_stack((
        np.arange(1, atom_types.shape[0] + 1), 
        atom_types, 
        positions
    ))
    
    return configuration


def calculate_masks(cutoff=7.5, voxel_size=1.62, grid_size=9):
    """Calculate mask indices for voxels outside the cutoff distance.
    
    Args:
        cutoff (float): Cutoff distance
        voxel_size (float): Size of each voxel
        grid_size (int): Size of the grid
    
    Returns:
        np.ndarray: Indices of masked voxels
    """
    center = grid_size // 2
    i, j, k = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        np.arange(grid_size), 
        indexing='ij'
    )

    i = i.astype(np.float32) * voxel_size
    j = j.astype(np.float32) * voxel_size
    k = k.astype(np.float32) * voxel_size
    center_pos = center * voxel_size

    dist = np.sqrt((i - center_pos)**2 + (j - center_pos)**2 + (k - center_pos)**2)
    mask = dist > cutoff
    mask_indices = np.argwhere(mask)

    return mask_indices

def get_cropped_pbc_region(grid, centers, crop_radius, masks):
    """Extract cropped regions around specified centers with periodic boundary conditions.
    
    Args:
        grid (np.ndarray): 3D grid array of atom types
        centers (np.ndarray): Array of center coordinates (N, 3)
        crop_radius (int): Radius for cropping
        masks (np.ndarray): Mask indices to apply
    
    Returns:
        np.ndarray: Cropped regions array
    """
    ncell = grid.shape[0]
    crop_size = 2 * crop_radius + 1

    # Create index ranges with periodic boundaries
    id_range = np.arange(-crop_radius, crop_radius + 1).reshape(1, -1, 1)
    id_range = (centers[:, np.newaxis, :] + id_range) % ncell
    ix, iy, iz = id_range[:, :, 0], id_range[:, :, 1], id_range[:, :, 2]

    # Broadcast indices for 3D indexing
    ii = ix[:, :, None, None]
    ii = np.broadcast_to(ii, (ii.shape[0], ii.shape[1], crop_size, crop_size))
    jj = iy[:, None, :, None]
    jj = np.broadcast_to(jj, (jj.shape[0], crop_size, jj.shape[2], crop_size))
    kk = iz[:, None, None, :]
    kk = np.broadcast_to(kk, (kk.shape[0], crop_size, crop_size, kk.shape[3]))
    
    out = grid[ii, jj, kk].copy()

    # Apply masks
    if masks.shape[0] > 0:
        out[:, masks[:, 0], masks[:, 1], masks[:, 2]] = 0
    out[out == -1] = 0
    
    return out

def first_neighbor_indices():
    """Get the indices of the first neighboring voxels in a 3D grid.

    Returns:
        np.ndarray: Array of shape (8, 3) containing the relative indices of first neighbors
    """
    indices = [
        [-1, -1, -1], [-1, -1, 1],
        [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1],
        [1, 1, -1], [1, 1, 1]
    ]
    return np.array(indices)

def rotate_and_mirror_data(data, crop_radius):
    """Apply rotations and mirroring to generate data for all first neighbor directions.
    
    Args:
        data (np.ndarray): Input data array of shape [N, D, D, D]
        crop_radius (int): Radius of the cropped region
    
    Returns:
        tuple: (cubes, directions, vacancy_mask)
    """
    N, D, _, _ = data.shape
    directions = first_neighbor_indices()
    cubes = np.empty((N, directions.shape[0], D, D, D), dtype=data.dtype)
    
    # Apply transformations for each direction
    # Direction 0: [-1, -1, -1] - Rotate 180° around z-axis + flip Z
    cubes[:, 0] = np.flip(np.rot90(data, k=2, axes=(1, 2)), axis=3)
    
    # Direction 1: [-1, -1, 1] - Rotate 180° around z-axis
    cubes[:, 1] = np.rot90(data, k=2, axes=(1, 2))
    
    # Direction 2: [-1, 1, -1] - Rotate 90° CCW around z-axis + flip Z
    cubes[:, 2] = np.flip(np.rot90(data, k=-1, axes=(1, 2)), axis=3)
    
    # Direction 3: [-1, 1, 1] - Rotate 90° CCW around z-axis
    cubes[:, 3] = np.rot90(data, k=-1, axes=(1, 2))
    
    # Direction 4: [1, -1, -1] - Rotate 90° CW around z-axis + flip Z
    cubes[:, 4] = np.flip(np.rot90(data, k=1, axes=(1, 2)), axis=3)
    
    # Direction 5: [1, -1, 1] - Rotate 90° CW around z-axis
    cubes[:, 5] = np.rot90(data, k=1, axes=(1, 2))
    
    # Direction 6: [1, 1, -1] - flip Z
    cubes[:, 6] = np.flip(data, axis=3)

    # Direction 7: [1, 1, 1] - no transformation needed
    cubes[:, 7] = data

    directions = directions.astype(np.int32)
    
    # Check if the jumping position is a vacancy
    vacancy_mask = (cubes[:, :, crop_radius + 1, crop_radius + 1, crop_radius + 1] == 0)

    # Flatten the last three dimensions
    cubes = cubes.reshape(N, directions.shape[0], -1)
    
    return cubes, directions, vacancy_mask

def count_pairs_vectorized(grid, n_types, directions):
    """Count pairs of atom types in specified directions using vectorized operations.
    
    Args:
        grid (np.ndarray): 3D grid of atom types
        n_types (int): Number of atom types
        directions (np.ndarray): Direction vectors for neighbor counting
    
    Returns:
        np.ndarray: Pair count matrix normalized by factor of 2
    """
    pair_mat = np.zeros((n_types, n_types), dtype=np.int64)
    grid_indices = grid - 1  # Convert to 0-based indexing

    for d in directions:
        shifted_grid_indices = np.roll(grid_indices, shift=d, axis=(0, 1, 2))
        
        valid_mask = (grid_indices >= 0) & (shifted_grid_indices >= 0)
        
        types1 = grid_indices[valid_mask]
        types2 = shifted_grid_indices[valid_mask]
        
        flat_indices = types1 * n_types + types2
        pair_counts = np.bincount(flat_indices, minlength=n_types * n_types)
        pair_mat += pair_counts.reshape((n_types, n_types))

    return pair_mat / 2.0

def cal_local_chemical_order(grid):
    """Calculate local chemical order parameters.

    Args:
        grid (np.ndarray): Input grid of atom types

    Returns:
        list: Local chemical order parameters for all pair types
    """
    directions = first_neighbor_indices()
    n_types = 3
    type_indices = np.array([1, 2, 3])  # Nb=1, Mo=2, Ta=3
    
    # Count atoms of each type
    types = grid[grid > 0]
    counts = np.array([(types == t).sum() for t in type_indices], dtype=np.float64)
    counts[counts == 0] = 1  # Avoid division by zero

    # Count pairs of atom types in first nearest neighbor shell
    pairs = count_pairs_vectorized(grid, n_types, directions)

    # Calculate normalized chemical order parameters
    norm_deltas = np.zeros(6)
    norm_deltas[0] = pairs[0, 0] / counts[0] * 2 - 8 / 3      # NbNb
    norm_deltas[1] = (pairs[0, 1] + pairs[1, 0]) / counts[0] - 8 / 3  # NbMo
    norm_deltas[2] = (pairs[0, 2] + pairs[2, 0]) / counts[0] - 8 / 3  # NbTa
    norm_deltas[3] = pairs[1, 1] / counts[1] * 2 - 8 / 3      # MoMo
    norm_deltas[4] = (pairs[1, 2] + pairs[2, 1]) / counts[1] - 8 / 3  # MoTa
    norm_deltas[5] = pairs[2, 2] / counts[2] * 2 - 8 / 3      # TaTa
    
    return norm_deltas.tolist()
