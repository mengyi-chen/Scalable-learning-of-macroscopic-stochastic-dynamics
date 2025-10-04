# Critical Behavior of the 2D Ising Model

This is the official implementation of our method on the Ising model to identify the critical behavior near the phase transition.

## Directory Structure

- **config/**
  - `config.yaml`: Configuration file containing default parameters for the simulations.

- **raw_data/**
  - `raw_data_generation.py`: microscopic simulation of the Ising model to generate trajectory data.
  - `glauber2d_ising.py`: Implementation of the Continuous-time Glauber dynamics for the Ising model.

- **raw_data_upsample/**
  - `upsampling_evolution.py`: Implementation of the hierarchical upsampling scheme and partial evolution scheme

- **train/**
  - `closure_modeling.py`: Script for training an autoencoder for identifying closure variables .
  - `train_SDE_partial.py`: Script for training stochastic differential equation models.

- **utils/**
  - Utility scripts for model building, data processing, and other helper functions.

- **test/**
  - Contains test scripts and visualizations for validating the model.
  - `test_ising_model.py`: Test the performance of the SDE model
  - `plot.ipynb`: visualization of the results


## Getting Started
To run the whole pipeline, simply execute:
```
bash run.sh
```
