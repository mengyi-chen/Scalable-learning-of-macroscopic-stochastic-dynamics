#!/bin/bash

# Step 1: Generate small-system trajectory distribution and test data
cd raw_data
echo "Generating small-system trajectory distribution and test data..."
python batch_solver.py  --nx 100 --bs 50
python batch_solver.py --nx 200 --bs 20

k_values=(0.05 0.1 0.15)
m_values=(0.45 0.5 0.55)

for k in "${k_values[@]}"; do
    for m in "${m_values[@]}"; do
        python batch_solver.py --nx 200 --bs 50 --flag_test --m $m --k $k
    done
done
cd ..

# Step 2: generate large-system snapshots from small-system trajectory distribution
cd raw_data_upsample
# hierarchical upsampling scheme 
echo "Generating large-system snapshots from small-system trajectory distribution..."
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
python train_SDE_partial.py --coeff 5 --seed 0

# coeff_list=(1 2.5 5 10 20)
# seed_list=(0 1 2)
# method=('ours', 'naive')

# for coeff in "${coeff_list[@]}"; do
#     for seed in "${seed_list[@]}"; do
#         for m in "${method[@]}"; do
#             python train_SDE_partial.py --coeff $coeff --seed $seed --method $m &
#         done
#     done
# done
cd ..

echo "Workflow completed successfully!"