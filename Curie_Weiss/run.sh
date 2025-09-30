#!/bin/bash

# Step 1: Generate small-system trajectory distribution and test data
cd raw_data
echo "Generating small-system trajectory distribution ..."
python raw_data_generation.py --L 8 --steps 200 --num_run 1000 --data_mode train # n_s = 8^2
python raw_data_generation.py --L 16 --steps 200 --num_run 1000 --data_mode train # n_s = 16^2
python raw_data_generation.py --L 32 --steps 200 --num_run 1000 --data_mode train # n_s = 32^2
python raw_data_generation.py --L 64 --steps 200 --num_run 1000 --data_mode train # n_s = 64^2

python raw_data_generation.py --L 64 --steps 200 --num_run 20 --data_mode val # validation data

echo "Generating test data for large system (n=64) with different initial magnetization..."
python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.5 --data_mode test
python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag -0.25 --data_mode test
python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag -0.5 --data_mode test
python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.25 --data_mode test
python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.0 --data_mode test
cd ..

# Step 2: generate large-system snapshots from small-system trajectory distribution
cd raw_data_upsample
# hierarchical upsampling scheme 
echo "Running hierarchical upsampling scheme and partial evolution scheme..."
echo "Generating large-system snapshots (n=64^2) from small-system trajectory distribution..."
python upsampling_evolution.py --patch_L 8 --steps 200 # n_s = 8^2
python upsampling_evolution.py --patch_L 16 --steps 200 # n_s = 16^2
python upsampling_evolution.py --patch_L 32 --steps 200 # n_s = 32^2
python upsampling_evolution.py --patch_L 64 --steps 200 # n_s = 64^2

cd ..

# Step 3: Closure modeling of macroscopic dynamics
cd train
# Learn the SDE model for macroscopic dynamics
echo "Learning the SDE model for macroscopic dynamics..."
python train_SDE_partial.py --patch_L 16 --method ours

# patch_L_list=(8 16 32 64)
# method=('ours', 'naive')

# for patch_L in "${patch_L_list[@]}"; do
#     for m in "${method[@]}"; do
#         python train_SDE_partial.py --patch_L $patch_L --method $m 
#     done
# done

cd ..

echo "Workflow completed successfully!"