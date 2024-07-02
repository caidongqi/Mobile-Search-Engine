import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

# Loop over lora_layers from 0 to 31


worker_num = 4
for i in range(1,7):
    if i%worker_num==0 and i!=0:
        command = f"python get_embedding_harsmart.py --lora_layers {i} --device cuda:0 " # stop to run this command
    else:
        command = f"python get_embedding_harsmart.py --lora_layers {i} --device cuda:{i%worker_num} &" # put it in the backend and run the next one concurrently
    logging.info(f"Running command: {command}")
    os.system(command)
