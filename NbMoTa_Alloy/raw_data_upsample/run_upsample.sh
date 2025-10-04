#!/bin/bash

# 8192 atoms
# for T in 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000; do
#     for seed in {0..9}; do
#         python upsampling_evolution.py --temperature $T --seed $seed --L 16 &
#     done
# done


for T in 1400 1600 1800 2000 2200 2400 2600 2800 3000; do
    for seed in {0..9}; do
        python upsampling_evolution.py --temperature $T --seed $seed --L 16 &
    done
done


# 65536 atoms 
# for T in 2000; do
#     for seed in {0..99}; do
#         python upsampling_evolution.py --temperature $T --seed $seed --L 32 --n_events 0 &
#     done
# done

# wait

# 524288 atoms
# for T in 2000; do
#     for seed in {0..99}; do
#         python upsampling_evolution.py --temperature $T --seed $seed --L 64 --n_events 0 &
#     done
# done

