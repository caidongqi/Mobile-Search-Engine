import os
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

for i in range(1, 33):
    command = f"python e2e_coco_lora_32.py --S 1 --N {i} --Q 10 --device cuda:0 --version layer32-{i}  >logs/coco-train/avg-layers32-{i}.log 2>&1"
    logging.info(f"Running command: {command}")
    os.system(command)
