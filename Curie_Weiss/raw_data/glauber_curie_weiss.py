from matplotlib.pylab import f
import numba
import random
import numpy as np
from numba import prange
import time, os 
from numba import njit
import numpy as np
from numba import njit

import numpy as np
from numba import njit

@njit
def _glauberCW(spin, L, beta, h=0.0, n_events=None):
    """
    Continuous-time Glauber dynamics for Curie–Weiss (complete graph) in 2D.
    """
    V = L * L
    if n_events is None:
        n_events = V

    spin_flat = spin.ravel()  # view
    # Buckets for +1 and -1 indices
    pos_idx = np.empty(V, dtype=np.int64)
    neg_idx = np.empty(V, dtype=np.int64)
    n_pos = 0
    n_neg = 0
    ssum = 0
    for i in range(V):
        s = spin_flat[i]
        ssum += s
        if s > 0:
            pos_idx[n_pos] = i; n_pos += 1
        else:
            neg_idx[n_neg] = i; n_neg += 1

    total_time = 0.0

    for _ in range(n_events):
        # Mean-field terms for the two classes
        R_plus  = (ssum - 1.0) / V
        R_minus = (ssum + 1.0) / V

        dH_plus  = 2.0 * ( R_plus + h)         # flip +1 -> -1
        dH_minus = 2.0 * (-R_minus - h)        # flip -1 -> +1

        # Logistic rates; clamp beta*dH if you worry about overflow
        r_plus  = 1.0 / (1.0 + np.exp(beta * dH_plus))
        r_minus = 1.0 / (1.0 + np.exp(beta * dH_minus))

        tot_rate = n_pos * r_plus + n_neg * r_minus
        if tot_rate <= 0.0:
            break

        # Exponential waiting time increment
        u = np.random.rand()
        if u < 1e-12:
            u = 1e-12
        total_time += -np.log(u) / tot_rate

        # Choose class and flip one site (swap-remove in bucket)
        if np.random.rand() < (n_pos * r_plus) / tot_rate:
            # Flip a +1 site
            j = np.random.randint(n_pos)
            idx = pos_idx[j]
            n_pos -= 1
            pos_idx[j] = pos_idx[n_pos]
            neg_idx[n_neg] = idx
            n_neg += 1
            spin_flat[idx] = -1
            ssum -= 2
        else:
            # Flip a -1 site
            j = np.random.randint(n_neg)
            idx = neg_idx[j]
            n_neg -= 1
            neg_idx[j] = neg_idx[n_neg]
            pos_idx[n_pos] = idx
            n_pos += 1
            spin_flat[idx] = +1
            ssum += 2

    return spin.reshape(L, L), total_time



class Glauber2DCW:
    def __init__(self, args):
        self.L = args.L
        self.steps = args.steps
        self.eqstep = args.steps // 2
        self.mcstep = args.steps // 2
        self.area = self.L ** 2  # 2D volume
        self.mag = args.mag

    def _init_spin(self):

        if self.mag == None:
            initial_mag = 2 * np.random.rand() - 1
            num_up = int((1 + initial_mag) * self.area / 2)
        else:
            num_up = int((1 + self.mag) * self.area / 2)
            
        # Create initial spin configuration
        num_down = self.area - num_up
        spins = np.concatenate([np.ones(num_up), -np.ones(num_down)])
        np.random.shuffle(spins)

        return spins.reshape(self.L, self.L)  # 2D reshape

    def _calc_energy(self, spin):
        mag = self._calc_magnetization(spin)
        Hamiltonian = - mag ** 2 / 2 - self.h * mag
        return Hamiltonian
    
    def _calc_magnetization(self, spin):
        return np.sum(spin) / self.area
    
    def _compute_domain_wall_density_2d(self, spin):
        """
        Compute domain wall density in a 2D CW model configuration.
        
        Args:
            spin (numpy.ndarray): 2D array of shape [L, L] with values -1 or +1.
            
        Returns:
            float: domain wall density
        """
        # Periodic shifts in x, y
        shift_x = np.roll(spin, 1, axis=0)
        shift_y = np.roll(spin, 1, axis=1)

        # Count domain walls
        dw_x = 0.5 * (1 - spin * shift_x)
        dw_y = 0.5 * (1 - spin * shift_y)

        total_dw = dw_x + dw_y
        rho_dw = total_dw.sum() / (2 * self.area)

        return rho_dw

    
    def simulate(self, beta, h=0, n_events=64):
        self.h = h
        E1, M1, E2, M2 = 0, 0, 0, 0
        M1_list = []
        config_list = []
        kmc_time = [0]
        spin = self._init_spin()

        for _ in range(self.eqstep):
            M = self._calc_magnetization(spin)
            M1_list.append(M)
            config_list.append(spin.copy())
            
            _, delta_t = _glauberCW(spin, self.L, beta, h=self.h, n_events=n_events)
            kmc_time.append(kmc_time[-1] + delta_t)

        # Monte Carlo steps
        for _ in range(self.mcstep):

            E = self._calc_energy(spin)
            M = self._calc_magnetization(spin)
            M1_list.append(M)
            config_list.append(spin.copy())

            M = np.abs(M)
            E1 += E / self.mcstep
            M1 += M / self.mcstep
            E2 += E ** 2 / self.mcstep
            M2 += M ** 2 / self.mcstep

            _, delta_t = _glauberCW(spin, self.L, beta, h=self.h, n_events=n_events)
            kmc_time.append(kmc_time[-1] + delta_t)

        config_list = np.array(config_list)
        return E1,  M1, (E2 - E1 * E1) * beta ** 2 * self.area, (M2 - M1 * M1) * beta * self.area, M1_list, config_list, kmc_time[:-1]