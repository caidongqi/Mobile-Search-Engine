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
for i in {29..1}; do
    commands+=("python get_embedding_flickr8k_text.py --lora_layers $i ")
done

# Run the commands in parallel batches of 5
run_batch "${commands[@]}"