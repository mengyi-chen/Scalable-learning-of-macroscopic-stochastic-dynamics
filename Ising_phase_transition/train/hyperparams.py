# Hyperparameter Patterns for Ising Phase Transition
# Format: T -> L -> {epsilon, coeff, epochs}

HYPERPARAMETERS = {
    # L = 16
    16: {
        2.25: {"epsilon": 1e-05, "coeff": 1.0, "epochs": 25},
        2.26: {"epsilon": 1e-05, "coeff": 1.0, "epochs": 25},
        2.27: {"epsilon": 1e-05, "coeff": 1.0, "epochs": 25},
        2.28: {"epsilon": 1e-05, "coeff": 1.0, "epochs": 50},
        2.29: {"epsilon": 1e-05, "coeff": 1.0, "epochs": 25},
    },
    
    # L = 32
    32: {
        2.25: {"epsilon": 0.0001, "coeff": 2.5, "epochs": 100},
        2.26: {"epsilon": 0.0001, "coeff": 2.5, "epochs": 100},
        2.27: {"epsilon": 0.0001, "coeff": 2.2, "epochs": 100},
        2.28: {"epsilon": 0.0001, "coeff": 2.2, "epochs": 100},
        2.29: {"epsilon": 0.0001, "coeff": 2.0, "epochs": 100},
    },
    
    # L = 48
    48: {
        2.25: {"epsilon": 0.0001, "coeff": 6.0, "epochs": 200},
        2.26: {"epsilon": 0.0001, "coeff": 6.0, "epochs": 200},
        2.27: {"epsilon": 0.0001, "coeff": 5.4, "epochs": 200},
        2.28: {"epsilon": 0.0001, "coeff": 5.0, "epochs": 200},
        2.29: {"epsilon": 0.0001, "coeff": 4.5, "epochs": 200},
    },
    
    # L = 64
    64: {
        2.25: {"epsilon": 0.0001, "coeff": 10.0, "epochs": 200},
        2.26: {"epsilon": 0.0001, "coeff": 9.0, "epochs": 100},
        2.27: {"epsilon": 0.0001, "coeff": 10.0, "epochs": 200},
        2.28: {"epsilon": 0.0001, "coeff": 8.0, "epochs": 200},
        2.29: {"epsilon": 0.0001, "coeff": 7.0, "epochs": 200},
    },
    
    # L = 128
    128: {
        2.27: {"epsilon": 0.0001, "coeff": 32.0, "epochs": 500},
    }
}

def get_hyperparameters(L, T):
    """
    Get hyperparameters for given lattice size L and temperature T.
    
    Args:
        L (int): Lattice size
        T (float): Temperature
        
    Returns:
        dict: Dictionary with epsilon, coeff, and epochs
        
    Raises:
        KeyError: If the (L, T) combination is not found
    """
    if L not in HYPERPARAMETERS:
        available_L = list(HYPERPARAMETERS.keys())
        raise KeyError(f"Lattice size L={L} not found. Available L values: {available_L}")
    
    if T not in HYPERPARAMETERS[L]:
        available_T = list(HYPERPARAMETERS[L].keys())
        raise KeyError(f"Temperature T={T} not found for L={L}. Available T values: {available_T}")
    
    return HYPERPARAMETERS[L][T].copy()

def print_hyperparameters():
    """Print all available hyperparameters in a formatted table."""
    print("Available Hyperparameter Combinations:")
    print("=" * 50)
    
    for L in sorted(HYPERPARAMETERS.keys()):
        print(f"\nL = {L}:")
        for T in sorted(HYPERPARAMETERS[L].keys()):
            params = HYPERPARAMETERS[L][T]
            print(f"  T={T}: epsilon={params['epsilon']}, coeff={params['coeff']}, epochs={params['epochs']}")

# Example usage
if __name__ == "__main__":
    print_hyperparameters()
    
    print(f"\nExample: Get hyperparameters for L=32, T=2.25")
    params = get_hyperparameters(32, 2.25)
    print(f"Parameters: {params}")