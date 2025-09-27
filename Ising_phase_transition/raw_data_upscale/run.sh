#!/bin/bash

# for T in 2.2 2.21 2.22 2.23 2.24 2.25 2.26 2.27 2.28 2.29 2.30; do
for T in 2.25 2.26 2.27 2.28 2.29; do
    # python main.py --L_small 16 --L_large 32 --T $T --gpu_idx 6 &
    python main.py --L_small 16 --L_large 48 --T $T --gpu_idx 6 &
    # python main.py --L_small 32 --L_large 64 --T $T --gpu_idx 6 &
    # python main.py --L_small 64 --L_large 128 --T $T --gpu_idx 6 --n_events 32 &
    # wait
done


