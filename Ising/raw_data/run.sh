#!/bin/bash

python raw_data_generation.py --L 8 --steps 200 --num_run 1000
python raw_data_generation.py --L 16 --steps 200 --num_run 1000
python raw_data_generation.py --L 32 --steps 200 --num_run 1000
python raw_data_generation.py --L 64 --steps 200 --num_run 1000

# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --flag_test

# python generate_labels.py --L 64 

# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.5
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.75
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag -0.25
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag -0.5
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag -0.75
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.25
# python raw_data_generation.py --L 64 --steps 500 --num_run 20 --mag 0.0

