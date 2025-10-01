#!/bin/bash

# Step 1: Generate small-system trajectory distribution and test data
cd raw_data
echo "Generating small-system (n_s = 100) trajectory distribution ..."
python batch_solver.py  --nx 100 --bs 50 --data_mode train

echo "Generating validation data for large system (n=200) ..."
python batch_solver.py --nx 200 --bs 20 --data_mode val

echo "Generating test data for large system (n=200) with various parameters..."
k_values=(0.05 0.1 0.15)
m_values=(0.45 0.5 0.55)
for k in "${k_values[@]}"; do
    for m in "${m_values[@]}"; do
        python batch_solver.py --nx 200 --bs 50 --data_mode test --m $m --k $k
    done
done
cd ..

# Step 2: generate large-system snapshots from small-system trajectory distribution
cd raw_data_upsample
# hierarchical upsampling scheme 
echo "Generating large-system snapshots (n=200) for training from small-system (n_s=100) trajectory distribution..."
python hierarchical_upsampling.py
# partial evolution scheme
echo "Running partial evolution scheme..."
python partial_evolution.py
cd ..

# Step 3: Closure modeling of macroscopic dynamics
cd train
# Train autoencoder for identifying closure variables
echo "Training autoencoder for identifying closure variables..."
python closure_modeling.py

# Learn the SDE model for macroscopic dynamics
echo "Learning the SDE model for macroscopic dynamics..."
# python train_SDE_partial.py --coeff 5 --seed 0

coeff_list=(1 2.5 5 10 20)
for coeff in "${coeff_list[@]}"; do
    python train_SDE_partial.py --coeff $coeff --seed $seed 
done
cd ..

echo "Workflow completed successfully!"