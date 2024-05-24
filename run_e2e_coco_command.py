import itertools
import json

# Define parameters
Ns = [2, 6, 12]
Qs = [5, 10, 20, 50, 100, 200, 500]
Ss = [1, 10, 30, 50, 60, 70, 80, 90, 100, 300, 400, 500]

# Create a list of all combinations of Ns, Qs, and Ss
combinations = [(N, S, Q) for Q in Qs for N, S in itertools.product(Ns, Ss)]

# Number of GPUs
num_gpus = 4

# Generate the commands in a round-robin fashion across the GPUs
commands = []
for i, (N, S, Q) in enumerate(combinations):
    device = f"cuda:{i % num_gpus}"
    log_file = f'logs/coco/S={S}_N={N}_Q={Q}.log'
    command = f"python coco_mobile-search_engine_lora.py --S {S} --N {N} --Q {Q} --device {device} > {log_file} 2>&1"
    commands.append(command)

# Output the commands as a list of strings
for command in commands:
    print(command)

# Save commands to a file
with open('commands.txt', 'w') as f:
    json.dump(commands, f)
    
print("Commands have been saved to 'commands.txt'")
