# Curie-Weiss Model

This is an implementation of our method on the Curie-Weiss model.

## Directory Structure

- **config/**
  - `config.yaml`: Configuration file containing default parameters for the simulations.

- **raw_data/**
  - `raw_data_generation.py`: microscopic simulation of the Curie-Weiss model to generate trajectory data.
  - `glauber_curie_weiss.py`: Implementation of the Continuous-time Glauber dynamics for the Curie-Weiss model.

- **raw_data_upsample/**
  - `upsampling_evolution.py`: Implementation of the hierarchical upsampling scheme and partial evolution scheme

- **train/**
  - `train_SDE_partial.py`: Script for training stochastic differential equation models.

- **utils/**
  - Utility scripts for model building, data processing, and other helper functions.

- **test/**
  - Contains test scripts and visualizations for validating the model.
  - `test.ipynb`: Jupyter notebook for testing the results
  - `plot.ipynb`: visualization of the results

## Getting Started
To run the whole pipeline, simply execute:
```
bash run.sh
```
