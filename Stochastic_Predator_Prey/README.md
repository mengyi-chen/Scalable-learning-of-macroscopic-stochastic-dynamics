# Stochastic Predator-Prey Model

This is an implementation of our method on the stochastic predator-prey system.

## Directory Structure

- **config/**
  - `config.yaml`: Configuration file containing default parameters for the simulations.

- **raw_data/**
  - `batch_solver.py`: microscopic simulation of the stochstic predator-prey system using numerical solvers.

- **raw_data_upsample/**
  - `hierarchical_upsampling.py`: generate large-system data distribution $\mathcal{D}$ from small-system trajectory distributions $\mathcal{D}_s$.
  - `partial_evolution.py`: Implementation of the partial evolution scheme to generate $x_{t+dt, \mathcal{I}}$.

- **train/**
  - `closure_modeling.py`: Script for training an autoencoder for identifying closure variables .
  - `train_SDE_partial.py`: Script for training stochastic differential equation models.

- **utils/**
  - Utility scripts for model building, data processing, and other helper functions.

- **test/**
  - Contains test scripts and visualizations for validating the model.
  - `test.ipynb`: Jupyter notebook for testing the results
  - `plot.py`: visualization of the results


## Getting Started
To run the whole pipeline, simply execute:
```
bash run.sh
```
