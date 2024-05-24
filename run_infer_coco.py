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

for i in range(1,33):
    command = f"python test_coco_origin.py --vision_num_blocks {i}"
    logging.info(f"Running command: {command}")
    os.system(command)
