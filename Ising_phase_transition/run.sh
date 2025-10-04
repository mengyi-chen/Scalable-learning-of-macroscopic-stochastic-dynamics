#!/bin/bash

# Step 1: Generate small-system trajectory distribution and test data
cd raw_data
echo "Generating small-system trajectory distribution ..."
python raw_data_generation.py --L 16 --steps 32000 --num_run 100 --n_proc 100 --T 2.27 --data_mode train
python raw_data_generation.py --L 64 --steps 32000 --num_run 50 --n_proc 50 --T 2.27 --data_mode val

# for T in 2.25 2.26 2.27 2.28 2.29; do
#     python raw_data_generation.py --L 16 --steps 32000 --num_run 100 --n_proc 100 --T $T --data_mode train 
# done

# for T in 2.25 2.26 2.27 2.28 2.29; do
#     python raw_data_generation.py --L 16 --steps 32000 --num_run 50 --n_proc 50 --T $T --data_mode val 
#     python raw_data_generation.py --L 32 --steps 32000 --num_run 50 --n_proc 50 --T $T --data_mode val
#     python raw_data_generation.py --L 48 --steps 32000 --num_run 50 --n_proc 50 --T $T --data_mode val
#     python raw_data_generation.py --L 64 --steps 32000 --num_run 50 --n_proc 50 --T $T --data_mode val
# done

# python raw_data_generation.py --L 128 --steps 32000 --num_run 50 --n_proc 50 --T 2.27 --data_mode val
cd ..

# Step 2: generate large-system snapshots from small-system trajectory distribution
cd raw_data_upsample
# hierarchical upsampling scheme and partial evolution scheme
echo "Running hierarchical upsampling scheme and partial evolution scheme..."
echo "Generating large-system snapshots from small-system trajectory distribution..."
python upsampling_evolution.py --patch_L 16 --L 64 --T 2.27

# for T in 2.25 2.26 2.27 2.28 2.29; do
#     python upsampling_evolution.py --patch_L 16 --L 16 --T $T 
#     python upsampling_evolution.py --patch_L 16 --L 32 --T $T 
#     python upsampling_evolution.py --patch_L 16 --L 48 --T $T 
#     python upsampling_evolution.py --patch_L 16 --L 64 --T $T  
# done
# python upsampling_evolution.py --patch_L 16 --L 128 --T 2.27 --n_events 32 

cd ..

# # Step 3: Closure modeling of macroscopic dynamics
cd train

# Train autoencoder for identifying closure variables
echo "Training autoencoder for identifying closure variables..."
python closure_modeling.py --L 64 --num_epoch 10 --T 2.27 

# for T in 2.25 2.26 2.27 2.28 2.29; do
#     python closure_modeling.py --L 16 --num_epoch 10 --T $T 
#     python closure_modeling.py --L 32 --num_epoch 10 --T $T 
#     python closure_modeling.py --L 48 --num_epoch 10 --T $T 
#     python closure_modeling.py --L 64 --num_epoch 10 --T $T 
# done
# python closure_modeling.py --L 128 --num_epoch 10 --T $T   

# Learn the SDE model for macroscopic dynamics
echo "Learning the SDE model for macroscopic dynamics..."
python train_SDE_partial.py --T 2.27 --L 64 --seed 0 --default_hyperparams 

# for T in 2.25 2.26 2.28 2.29; do
#     for L in 16 32 48 64; do
#         for seed in 0 1 2 3 4; do
#             python train_SDE_partial.py --T $T --L $L --seed $seed --default_hyperparams &
#         done
#         wait
#     done
# done
# thon train_SDE_partial.py --T 2.27 --L 128 --seed $seed --default_hyperparams 

cd ..

echo "Workflow completed successfully!"