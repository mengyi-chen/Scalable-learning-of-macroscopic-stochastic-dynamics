#!/bin/bash

# python pretrain_AE_partial.py --box_L 64 --gpu_idx 0 &
# python pretrain_AE_partial.py --box_L 32 --gpu_idx 0 &
# python pretrain_AE_partial.py --box_L 16 --gpu_idx 1 &
# python pretrain_AE_partial.py --box_L 8 --gpu_idx 1 &

# python train_SDE_partial.py --box_L 64 --gpu_idx 0 &
# python train_SDE_partial.py --box_L 32 --gpu_idx 0 &
# python train_SDE_partial.py --box_L 16 --gpu_idx 1 &
# python train_SDE_partial.py --box_L 8 --gpu_idx 1 &

python train_SDE_partial.py --box_L 64 --method naive --gpu_idx 0 &
python train_SDE_partial.py --box_L 32 --method naive --gpu_idx 0 &
python train_SDE_partial.py --box_L 16 --method naive --gpu_idx 1 &
python train_SDE_partial.py --box_L 8 --method naive --gpu_idx 1 &
