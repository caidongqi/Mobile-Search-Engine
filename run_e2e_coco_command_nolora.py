import itertools
import json

# Define parameters
Ns = [6,7,8,9]
Qs = [5, 10, 20,30]
Ss = [30]

# Create a list of all combinations of Ns, Qs, and Ss
combinations = [(N,Q, S) for Q in Qs for N, S in itertools.product(Ns, Ss)]

# Number of GPUs
num_gpus = 1

# Generate the commands in a round-robin fashion across the GPUs
commands = []
for i, (N, Q,S) in enumerate(combinations):
    device = f"cuda:{i % num_gpus}"
    #version='val_model_v1' #with new lora predict models 
    #log_file = f'logs/ground_truth_val/e2e-val-S={S}_N={N}_Q={Q}_{version}.log'
    command = f"python e2e_coco_lora_32_max.py --S {S} --Q {Q} --N {N}  "
    commands.append(command)

# Output the commands as a list of strings
for command in commands:
    print(command)

# Save commands to a file
with open('commands_max.txt', 'w') as f:
    json.dump(commands, f)
    
print("Commands have been saved to 'commands_true.txt'")
