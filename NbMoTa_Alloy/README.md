# NbMoTa Alloy

This is the official implementation of our method on the NbMoTa alloy system.

## Directory Structure

### **raw_data/**
- `raw_data_generation.py`: kinetic Monte Carlo simulations of small NbMoTa alloy systems with 1024 atoms 
- `pipeline_1024.ipynb`: Data processing pipeline for 1024-atom simulations

### **raw_data_upsample/**
- `upsampling_evolution.py`: hierarchical upsampling scheme for generating large-system data distributions

###  **train/**
- `train_SDE_partial.py`: Stochastic differential equation learning with partial observations and multi-temperature training for alloy dynamics

### **utils/**
- Utility scripts for model building, data processing, and other helper functions.

### **test/**
- `test.ipynb`: Interactive analysis notebook for model validation

## Quick Start

**Generate small-system (1024 atoms) trajectory distribution:**
```bash
cd raw_data
bash run_1024.sh 
```
run `pipeline_1024.ipynb` to process the raw trajectory data

**Hierarchical upsampling scheme:**
```bash
cd raw_data_upsample
bash run_upsample.sh
```
run `pipeline_upsample.ipynb` to process the upsampled trajectory data

**Learn stochastic macroscopic dynamics:**
```bash
python train_SDE_partial.py --coeff 8 --N_atoms 8192
```

**Analyze results:**
```bash
cd test
```
run  `test.ipynb` to analyze the results

## Acknowledgments

This code is based on the github repository https://github.com/UCICaoLab/NKK.git and the following work
  
Xing, Bin, et al. "Neural network kinetics for exploring diffusion multiplicity and chemical ordering in compositionally complex materials." *Nature Communications* 15.1 (2024): 3879.