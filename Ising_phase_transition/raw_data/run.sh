#!/bin/bash

# for T in 2.2 2.21 2.22 2.23 2.24 2.25 2.26 2.27 2.28 2.29; do
for T in 2.25 2.26 2.27 2.28 2.29; do
    # python main.py --L 16 --steps 32000 --num_run 100 --n_proc 100 --T $T &
    # python main.py --L 32 --steps 32000 --num_run 50 --n_proc 50 --T $T &
    # python main.py --L 48 --steps 32000 --num_run 50 --n_proc 50 --T $T &
    # python main.py --L 64 --steps 32000 --num_run 50 --n_proc 50 --T $T &
    # python main.py --L 128 --steps 32000 --num_run 50 --n_proc 50 --T $T &
    # wait

    python generate_partial.py --T $T --L 16 --gpu_idx 5 &
    # python generate_partial.py --T $T --L 64 --gpu_idx 7 &
    # python generate_partial.py --T $T --L 32 --gpu_idx 7 &
    # python generate_partial.py --T $T --L 48 --gpu_idx 7 &
    # python generate_partial.py --T $T --L 128 --gpu_idx 7 &
    # wait
done
