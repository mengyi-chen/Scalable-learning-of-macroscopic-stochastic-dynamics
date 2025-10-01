#!/bin/bash

# for T in 2.2 2.21 2.22 2.23 2.24 2.25 2.26 2.27 2.28 2.29; do
# for T in 2.25 2.26 2.27 2.28 2.29; do
    # python closure_modeling.py --L 16 --gpu_idx 7 --epoch 10 --T $T &
    # python closure_modeling.py --L 32 --gpu_idx 6 --epoch 10 --T $T &
    # python closure_modeling.py --L 48 --gpu_idx 6 --epoch 10 --T $T &
    # python closure_modeling.py --L 64 --gpu_idx 7 --epoch 0 --T $T &
    # python closure_modeling.py --L 128 --gpu_idx 6 --epoch 0 --T $T &
    # wait 

    # python train_SDE_partial.py --patch_L 16 --L 16 --gpu_idx 7 --coeff 1 --num_epoch 25 --T $T &
   
# done



# python train_SDE_partial.py --T 2.27 --patch_L 16 --L 64 --gpu_idx 7 --coeff 10.5 --num_epoch 100 --seed 0 &
# python train_SDE_partial.py --T 2.27 --patch_L 16 --L 64 --gpu_idx 7 --coeff 10.5 --num_epoch 100 --seed 1 &
# python train_SDE_partial.py --T 2.27 --patch_L 16 --L 64 --gpu_idx 7 --coeff 10.5 --num_epoch 100 --seed 2 &
# python train_SDE_partial.py --T 2.27 --patch_L 16 --L 64 --gpu_idx 7 --coeff 10.5 --num_epoch 100 --seed 3 &
# python train_SDE_partial.py --T 2.27 --patch_L 16 --L 64 --gpu_idx 7 --coeff 10.5 --num_epoch 100 --seed 4 &


python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 6 --coeff 5.5 --num_epoch 200 --seed 0 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 6 --coeff 5.5 --num_epoch 200 --seed 1 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 6 --coeff 5.5 --num_epoch 200 --seed 2 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 6 --coeff 5.5 --num_epoch 200 --seed 3 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 6 --coeff 5.5 --num_epoch 200 --seed 4 &


python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 7 --coeff 6 --num_epoch 200 --seed 0 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 7 --coeff 6 --num_epoch 200 --seed 1 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 7 --coeff 6 --num_epoch 200 --seed 2 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 7 --coeff 6 --num_epoch 200 --seed 3 &
python train_SDE_partial.py --T 2.26 --patch_L 16 --L 48 --gpu_idx 7 --coeff 6 --num_epoch 200 --seed 4 &

wait 
