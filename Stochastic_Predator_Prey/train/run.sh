#!/bin/bash

coeff_list=(1 2.5 5 10 20)
for coeff in "${coeff_list[@]}"; do
    python train_SDE_partial.py --coeff $coeff --seed 2 &
done