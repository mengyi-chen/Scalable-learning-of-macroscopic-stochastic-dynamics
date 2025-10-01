import numba
import random
import numpy as np
from numba import prange
import time, os 

@numba.jit(nopython=True)
def _glauber_discrete(spin, L, beta, h=0):
    """Perform one sweep of Discrete-time Glauber dynamics for the Ising model.

    Args:
        spin (numpy.ndarray): 2D array of shape [L, L] with values -1 or +1.
        L (int): Length of the lattice.
        beta (float): Inverse temperature (1/k_B T).
        h (float, optional): External magnetic field strength. Defaults to 0.0. 

    Returns:
        numpy.ndarray: Updated spin configuration.
        float: Time increment for the KMC simulation.
    """

    for x in range(L):
        for y in range(L):
            s = spin[x, y]

            # Sum of the spins of nearest neighbors
            xpp = (x + 1) if (x + 1) < L else 0
            ypp = (y + 1) if (y + 1) < L else 0
            xnn = (x - 1) if (x - 1) >= 0 else (L - 1)
            ynn = (y - 1) if (y - 1) >= 0 else (L - 1)
            R = spin[xpp, y] + spin[x, ypp] + spin[xnn, y] + spin[x, ynn]

            # Check Metropolis-Hastings algorithm for more details
            dH = 2 * s * R + 2 * h * s  
            prob = 1.0 / (1.0 + np.exp(beta * dH))
            if np.random.rand() < prob:
                spin[x, y] *= -1  # Flip the spin


class Glauber2DIsing:
    """Glauber dynamics for 2D Ising model with periodic boundary conditions.
    """
    def __init__(self, args):
        self.L = args.L
        self.steps = args.steps
        self.eqstep = args.steps // 2
        self.mcstep = args.steps // 2
        self.area = self.L ** 2
        self.mag = args.mag

    def _init_spin(self):
        """Initialize the spin configuration.

        Returns:
            np.ndarray: 2D array representing the initial spin configuration.
        """
        
        if self.mag == None:
            # Random initialization
            initial_mag = 2 * np.random.rand() - 1
            num_up = int((1 + initial_mag) * self.area / 2)
        else:
            num_up = int((1 + self.mag) * self.area / 2)
            
        # Create initial spin configuration
        num_down = self.area - num_up
        spins = np.concatenate([np.ones(num_up), -np.ones(num_down)])
        np.random.shuffle(spins)

        return spins.reshape(self.L, self.L)  # 2D reshape

    # Calculate energy using neighbors
    def _calc_energy(self, spin):
        """Calculate the energy of the system.

        Args:
            spin (numpy.ndarray): 2D array of shape [L, L] with values -1 or +1.

        Returns:
            float: energy
        """
        R = np.roll(spin, 1, axis=0) + np.roll(spin, -1, axis=0) \
            + np.roll(spin, 1, axis=1) + np.roll(spin, -1, axis=1)
        Hamiltonian = np.sum(-R * spin) / (2 * self.area) - self.h * np.sum(spin) / self.area
        return Hamiltonian
    
    def _calc_magnetization(self, spin):
        """Calculate the magnetization of the system.

        Args:
            spin (numpy.ndarray): 2D array of shape [L, L] with values -1 or +1.

        Returns:
            float: magnetization
        """
        return np.sum(spin) / self.area

    def _compute_domain_wall_density_2d(self, spin):
        """
        Compute domain wall density in a 2D Ising model configuration.
        
        Args:
            spin (torch.Tensor): 2D tensor of shape [L, L] with values -1 or +1.
            
        Returns:
            float: domain wall density
        """
        # Periodic shifts in x, y, z
        shift_x = np.roll(spin, 1, axis=0)
        shift_y = np.roll(spin, 1, axis=1)

        # Count domain walls
        dw_x = 0.5 * (1 - spin * shift_x)
        dw_y = 0.5 * (1 - spin * shift_y)

        total_dw = dw_x + dw_y
        rho_dw = total_dw.sum() / (2 * self.area)

        return rho_dw.item()

    
    def simulate(self, beta, h=0):
        """Run the Glauber dynamics simulation.

        Args:
            beta (float): Inverse temperature (1/k_B T).
            h (int, optional): External magnetic field strength.

        Returns:
            float: Energy, magnetization, specific heat, and susceptibility.
            list: Magnetization time series
            np.ndarray: Configuration time series
        """

        self.h = h
        E1, M1, E2, M2 = 0, 0, 0, 0
        M1_list = []
        config_list = []
        spin = self._init_spin()

        for _ in range(self.eqstep):
            _glauber_discrete(spin, self.L, beta, h=self.h)
            M = self._calc_magnetization(spin)
            M1_list.append(M)
            config_list.append(spin.copy())
            
        # Monte Carlo steps
        for _ in range(self.mcstep):
            _glauber_discrete(spin, self.L, beta, h=self.h)
            E = self._calc_energy(spin)
            M = self._calc_magnetization(spin)
            M1_list.append(M)
            config_list.append(spin.copy())

            M = np.abs(M)
            E1 += E / self.mcstep
            M1 += M / self.mcstep
            E2 += E ** 2 / self.mcstep
            M2 += M ** 2 / self.mcstep
            
        config_list = np.array(config_list)
        return E1,  M1, (E2 - E1 * E1) * beta ** 2 * self.area, (M2 - M1 * M1) * beta * self.area, M1_list, config_list