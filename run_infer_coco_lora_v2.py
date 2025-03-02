import os
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

# print("Starting to sleep 30mins for waiting the end of embedding extraction")
# time.sleep(1800)
# print("Finished sleeping")

worker_num = 5
bad_i = [5, 7, 8, 9]
for i in bad_i:

    command = f"python test_coco_lora.py --vision_num_blocks {i} > logs/coco-train/infer-{i}.log 2>&1 &" # put it in the backend and run the next one concurrently
    logging.info(f"Running command: {command}")
    os.system(command)
