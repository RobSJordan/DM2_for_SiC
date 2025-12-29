#!/bin/bash

# This script runs multiple batches of simulations with different settings.
# It takes arguments in triplets:
# <number_of_runs> <number_of_atoms> <cooling_rate>
#
# You can provide multiple triplets to run different batches sequentially.
#
# Usage: ./run_multi_batch.sh <runs1> <atoms1> <rate1> [<runs2> <atoms2> <rate2> ...]
# Example: ./run_multi_batch.sh 5 3000 100 5 3000 10 5 3000 1

if [ "$#" -eq 0 ] || [ $(($# % 3)) -ne 0 ]; then
    echo "Usage: $0 <runs1> <atoms1> <rate1> [<runs2> <atoms2> <rate2> ...]"
    echo "Please provide arguments in triplets (number of runs, number of atoms, cooling rate)."
    exit 1
fi

while [ "$#" -gt 0 ]; do
    NUM_RUNS=$1
    NUM_ATOMS=$2
    COOL_RATE=$3
    shift 3

    echo "===================================================="
    echo "--- Starting Batch: $NUM_RUNS runs with $NUM_ATOMS atoms at $COOL_RATE K/ps ---"
    echo "===================================================="

    for i in $(seq 1 $NUM_RUNS)
    do
      ./run_workflow.py --natoms "$NUM_ATOMS" --rate "$COOL_RATE"
    done
done

echo "All simulation batches completed successfully."