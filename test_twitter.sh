#!/bin/bash

# Function to run the commands in parallel batches of 5
run_batch() {
    local batch=()
    for cmd in "$@"; do
        batch+=("$cmd")
        if [ ${#batch[@]} -eq 6 ]; then
            for b in "${batch[@]}"; do
                eval "$b" &
            done
            wait
            batch=()
        fi
    done

    # Run any remaining commands in the batch
    for b in "${batch[@]}"; do
        eval "$b" &
    done
    wait
}

# Collect all the commands
commands=()
for i in {32..1}; do
    for j in {24..1}; do
        commands+=("python test_twitter.py --vision_num_blocks $i --text_num_blocks $j")
    done
done

# Run the commands in parallel batches of 5
run_batch "${commands[@]}"