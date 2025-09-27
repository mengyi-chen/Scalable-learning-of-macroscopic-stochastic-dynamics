#!/bin/bash

# for T in 300 400 500 600 700 800 900 1000 1200; do
#     for seed in {0..9}; do
#         python upscale.py --temperature $T --seed $seed &
#     done
# done

# for T in 1400 1600 1800 2000 2200 2400 2600 2800 3000; do
#     for seed in {0..9}; do
#         python upscale.py --temperature $T --seed $seed &
#     done
# done

# for T in 2000; do
#     for seed in {0..99}; do
#         python upscale.py --temperature $T --seed $seed --box_L 32 --n_events 0 &
#     done
# done

# wait


for T in 2000; do
    for seed in {0..99}; do
        python upscale.py --temperature $T --seed $seed --box_L 64 --n_events 0 &
    done
done