# Scalable Learning of Macroscopic Stochastic Dynamics

This repository contains implementations and experiments for *Scalable learning of macroscopic stochastic dynamics*. The project is organized into multiple subdirectories, each focusing on different models and datasets.

## Directory Structure
### `Stochastic_Predator_Prey/`
Focuses on the stochastic predator-prey model.
- `raw_data/`: Scripts for generating raw data.
- `raw_data_upscale/`: Scripts for upscaling raw data.
- `test/`: Test scripts and data.
- `train/`: Training scripts for the model.
- `utils/`: Utility functions and modules.


### `Curie_Weiss/`
Contains scripts and data for the Curie-Weiss model.
- `raw_data/`: Scripts for generating raw data.
- `raw_data_upscale/`: Scripts for upscaling raw data and visualizations.
- `test/`: Test scripts and notebooks.
- `train/`: Training scripts for the model.
- `utils/`: Utility functions and modules.

### `Ising/`
Contains scripts and data for the Ising model.
- `raw_data/`: Scripts for generating raw data.
- `raw_data_upscale/`: Scripts for upscaling raw data and visualizations.
- `test/`: Test scripts, results, and visualizations.
- `train/`: Training scripts for the model.
- `utils/`: Utility functions and modules.

### `Ising_phase_transition/`
Focuses on the Ising phase transition.
- `raw_data/`: Scripts for generating raw data.
- `raw_data_upscale/`: Scripts for upscaling raw data and visualizations.
- `test/`: Test scripts, results, and visualizations.
- `train/`: Training scripts for the model.
- `utils/`: Utility functions and modules.

### `NbMoTa_Alloy/`
Contains scripts and data for NbMoTa alloy simulations.
- `input/`: Input data.
- `raw_data/`: Scripts for generating raw data.
- `raw_data_upscale/`: Scripts for upscaling raw data.
- `test/`, `train/`, `utils/`: Similar structure as other models.


## Key Files

- `requirements.txt`: Lists the Python dependencies for the project.
- `LICENSE`: License information for the repository.

## Getting Started
### Dependencies
* CUDA Version: 12.7
* GPU Model: NVIDIA L40S  


### Prerequisites
- Python 3.8 or higher
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mengyi-chen/Scalable-learning-of-macroscopic-stochastic-dynamics.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Scalable-learning-of-macroscopic-stochastic-dynamics
   ```
3. Install dependencies:
   ```bash
   conda create -n newenv python=3.9.21
   conda activate newenv
   pip install -r requirements.txt
   ```

### Running the Code
[TBD]
Each subdirectory contains its own `run.sh` scripts for executing experiments. For example, to run the stochastic predator-prey model:
```bash
cd Stochastic_Predator_Prey/raw_data
bash run.sh
```

