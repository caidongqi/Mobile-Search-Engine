#!/bin/bash

#睡眠1小时
# sleep 1h

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
for i in {32..24}; do
    for j in {32..1}; do
        commands+=("python test_harsmart_lora_without_head.py --vision_num_blocks $i --text_num_blocks $j")
    done
done

# Run the commands in parallel batches of 5
run_batch "${commands[@]}"