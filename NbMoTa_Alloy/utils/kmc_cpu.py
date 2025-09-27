
import numpy as np

class KMC_CPU():

    def __init__(self, attempt_frequency, boltzmann_constant, temperature, time_scale):

        self.attempt_frequency = attempt_frequency
        self.boltzmann_constant = boltzmann_constant      
        self.temperature = temperature
        self.time_scale = time_scale

    def execute(self, config_grid, pair_information, vacancy_indices):
        # config_grid: [2L, 2L, 2L] numpy array
        # vacancy_indices: [N, 3]
        directions, barriers = pair_information  # directions: (D, 3), barriers: (N, 8)

        # Compute jump rates
        rates = self.attempt_frequency * np.exp(-barriers / (self.boltzmann_constant * self.temperature)) # [N, 8]
        rates_flat = rates.flatten()  # [N * 8]
        total_rate = np.sum(rates_flat)
        # print('rates_flat:', rates_flat)

        probs = rates_flat / total_rate

        # Sample one jump
        rand = np.random.rand(1)[0]
        cum_probs = np.cumsum(probs)
        selected_flat_idx = np.searchsorted(cum_probs, rand)
        selected_flat_idx = min(selected_flat_idx, len(cum_probs) - 1)  # Ensure index is within bounds
        # print('index:', selected_flat_idx, 'cumulative_rates:', cum_probs)

        vacancy_idx = selected_flat_idx // 8
        direction_idx = selected_flat_idx % 8

        
        selected_direction = directions[direction_idx]

        # Compute time increment
        random_number = np.random.rand(1)[0]
        jumping_time = - np.log(random_number) / total_rate * self.time_scale

        # Update grid: swap vacancy and neighbor
        old_pos = vacancy_indices[vacancy_idx].copy()  # Make a copy to avoid confusion
        new_pos = (old_pos + selected_direction) % config_grid.shape[0] 
        neighbor_type = config_grid[new_pos[0], new_pos[1], new_pos[2]]
        config_grid[old_pos[0], old_pos[1], old_pos[2]] = neighbor_type
        config_grid[new_pos[0], new_pos[1], new_pos[2]] = 0
    
        # Update vacancy_indices
        vacancy_indices[vacancy_idx] = new_pos

        return (old_pos, new_pos, neighbor_type), jumping_time, config_grid, vacancy_indices


