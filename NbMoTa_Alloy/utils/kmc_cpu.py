
import numpy as np

class KMC_CPU:
    """CPU-based Kinetic Monte Carlo simulator for vacancy diffusion.
    """
    
    def __init__(self, attempt_frequency, boltzmann_constant, temperature, time_scale):
        """Initialize the KMC simulator with physical parameters.
        
        Args:
            attempt_frequency (float): Vibrational attempt frequency for jumps
            boltzmann_constant (float): Boltzmann constant in appropriate units
            temperature (float): Simulation temperature in Kelvin
            time_scale (float): Conversion factor for time units
        """
        self.attempt_frequency = attempt_frequency
        self.boltzmann_constant = boltzmann_constant
        self.temperature = temperature
        self.time_scale = time_scale

    def execute(self, config_grid, pair_information, vacancy_indices):
        """Execute one KMC step by selecting and performing an atomic jump.
           
        Args:
            config_grid (np.ndarray): 3D grid of atom types, shape [2L, 2L, 2L]
            pair_information (tuple): (directions, barriers) where:
                - directions (np.ndarray): Jump directions, shape (8, 3)
                - barriers (np.ndarray): Energy barriers, shape (N, 8)
            vacancy_indices (np.ndarray): Vacancy positions, shape [N, 3]
        
        Returns:
            tuple: Contains:
                - jump_info (tuple): (old_pos, new_pos, neighbor_type)
                - jumping_time (float): Time increment for this jump
                - config_grid (np.ndarray): Updated configuration grid
                - vacancy_indices (np.ndarray): Updated vacancy positions
        """
        directions, barriers = pair_information
        
        # Validate input
        if barriers.size == 0:
            raise ValueError("No energy barriers provided")
        
        # Calculate transition rates using Arrhenius equation: k = ν₀ * exp(-E/kT)
        rates = self.attempt_frequency * np.exp(
            -barriers / (self.boltzmann_constant * self.temperature)
        )  # Shape: [N, 8]
        
        # Flatten rates for easier sampling
        rates_flat = rates.flatten()  # Shape: [N * 8]
        total_rate = np.sum(rates_flat)
        
        if total_rate <= 0:
            raise ValueError("Total transition rate is zero or negative")
        
        # Calculate probabilities for each possible jump
        probs = rates_flat / total_rate
        
        # Stochastic selection of jump using rejection-free algorithm
        rand = np.random.rand()
        cum_probs = np.cumsum(probs)
        selected_flat_idx = np.searchsorted(cum_probs, rand)
        selected_flat_idx = min(selected_flat_idx, len(cum_probs) - 1)
        
        # Decode selected jump: which vacancy and which direction
        vacancy_idx = selected_flat_idx // 8
        direction_idx = selected_flat_idx % 8
        selected_direction = directions[direction_idx]
        
        # Calculate time increment using exponential distribution
        random_number = np.random.rand()
        jumping_time = -np.log(random_number) / total_rate * self.time_scale
        
        # Execute the jump: update configuration grid
        old_pos = vacancy_indices[vacancy_idx].copy()
        new_pos = (old_pos + selected_direction) % config_grid.shape[0]
        
        # Get the type of atom that will jump into the vacancy
        neighbor_type = config_grid[new_pos[0], new_pos[1], new_pos[2]]
        
        # Swap vacancy and neighboring atom
        config_grid[old_pos[0], old_pos[1], old_pos[2]] = neighbor_type
        config_grid[new_pos[0], new_pos[1], new_pos[2]] = 0  # 0 represents vacancy
        
        # Update vacancy position tracking
        vacancy_indices[vacancy_idx] = new_pos
        
        # Return jump information and updated state
        jump_info = (old_pos, new_pos, neighbor_type)
        return jump_info, jumping_time, config_grid, vacancy_indices


