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

for i in [1,10, 30, 50, 60,70,80,90,100,200,300,400,500,600]:
    for j in range(1,33):
        command = f"python coco_model3_origin.py --S {i} --layer_num {j}"
        logging.info(f"Running command: {command}")
        os.system(command)
