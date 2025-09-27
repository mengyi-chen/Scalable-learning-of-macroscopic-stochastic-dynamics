#!/bin/bash

# Common settings
seeds=({10..19})
NUM_CORES=90

echo "=== RUNNING CONFIGURATION 1: Lower temperatures (300-1200) with 20M steps ==="
temperatures=(300 400 500 600 700 800 900 1000 1200)
total_jobs=$(( ${#temperatures[@]} * ${#seeds[@]} ))
echo "Running $total_jobs jobs (${#temperatures[@]} temperatures x ${#seeds[@]} seeds), up to $NUM_CORES in parallel."

# --- Generate all combinations and pipe to xargs ---
( # Start a subshell to group the output of the loops
for T in "${temperatures[@]}"; do
    for seed in "${seeds[@]}"; do
        printf "%s %s\n" "$T" "$seed"
    done
done
) | xargs -n 2 -P "$NUM_CORES" -I '{}' \
bash -c '
    # xargs reads two items per line and passes them as a single string.
    # We use `read` to split this string back into two variables.
    read -r T seed <<< "$1"

    # Construct the output directory path for this specific combination
    output_dir="../data/output_atoms_1024_steps_2000000/T_${T}_seed_${seed}"

    # Print a status message to know which job is starting
    echo "Starting job for T = ${T}, seed = ${seed}..."

    # Execute the actual Python command, quoting variables
    python main.py --temperature "${T}" \
    --output_dir "${output_dir}" --seed "${seed}" \
    --number_of_steps 2000000 \
    --box_L 8 \
    --number_of_log_steps 1000 \
    --number_of_dump_steps 1000 \

    echo "Finished job for T = ${T}, seed = ${seed}."
' bash {} # The placeholder {} receives the line "T seed"

# echo "=== FINISHED CONFIGURATION 1 ==="
# echo ""
# echo "=== RUNNING CONFIGURATION 2: Higher temperatures (1400-3000) with 2M steps ==="

# temperatures=(1400 1600 1800 2000 2200 2400 2600 2800 3000)
# total_jobs=$(( ${#temperatures[@]} * ${#seeds[@]} ))
# echo "Running $total_jobs jobs (${#temperatures[@]} temperatures x ${#seeds[@]} seeds), up to $NUM_CORES in parallel."

# # --- Generate all combinations and pipe to xargs ---
# ( # Start a subshell to group the output of the loops
# for T in "${temperatures[@]}"; do
#     for seed in "${seeds[@]}"; do
#         printf "%s %s\n" "$T" "$seed"
#     done
# done
# ) | xargs -n 2 -P "$NUM_CORES" -I '{}' \
# bash -c '
#     # xargs reads two items per line and passes them as a single string.
#     # We use `read` to split this string back into two variables.
#     read -r T seed <<< "$1"

#     # Construct the output directory path for this specific combination
#     output_dir="../data/output_atoms_1024_steps_2000000/T_${T}_seed_${seed}"

#     # Print a status message to know which job is starting
#     echo "Starting job for T = ${T}, seed = ${seed}..."

#     # Execute the actual Python command, quoting variables
#     python main.py --temperature "${T}" \
#     --output_dir "${output_dir}" --seed "${seed}" \
#     --number_of_steps 2000000 \
#     --box_L 8 \
#     --number_of_log_steps 1000 \
#     --number_of_dump_steps 1000 \

#     echo "Finished job for T = ${T}, seed = ${seed}."
# ' bash {} # The placeholder {} receives the line "T seed"

# echo "=== FINISHED CONFIGURATION 2 ==="
# echo "All jobs have been completed!"
