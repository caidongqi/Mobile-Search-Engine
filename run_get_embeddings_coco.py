import os

# Loop over lora_layers from 0 to 31
for i in range(32):
    command = f"python image_trunks_coco.py --lora_layers {i} --device cuda:0"
    print(f"Running command: {command}")
    os.system(command)
