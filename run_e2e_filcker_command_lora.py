import itertools
import json

# Define parameters
Ns = [8]
Qs = [1, 2, 5, 10, 20, 30, 40, 50, 60,70,80,90,100]
Ss = [1,10,40]

# Create a list of all combinations of Ns, Qs, and Ss
combinations = [(N, S, Q) for Q in Qs for N, S in itertools.product(Ns, Ss)]

# Number of GPUs
num_gpus = 1

# Generate the commands in a round-robin fashion across the GPUs
commands = []
for i, (N, S, Q) in enumerate(combinations):
    device = f"cuda:{i % num_gpus}"
    #version='val_model_v1' #with new lora predict models 
    #log_file = f'logs/ground_truth_val/e2e-val-S={S}_N={N}_Q={Q}_{version}.log'
    command = f"python e2e_flicker_model.py --S {S} --N {N} --Q {Q} "
    commands.append(command)

# Output the commands as a list of strings
for command in commands:
    print(command)

# Save commands to a file
with open('commands_flicker.txt', 'w') as f:
    json.dump(commands, f)
    
print("Commands have been saved to 'commands_true.txt'")
