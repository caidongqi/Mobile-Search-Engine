import os
import time
import logging
import time

# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间
local_time = time.localtime(timestamp)
# 格式化本地时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'logs/run/infer_{formatted_time}.log')], 
                    force=True)

worker_num = 4
for i in range(1,33):
    if i%(worker_num-1)==0 and i!=0:
        command = f"python get_layers_confidents.py --vision_num_blocks {i} > logs/infer/infer-{i}.log 2>&1" # stop to run this command
    else:
        command = f"python get_layers_confidents.py --vision_num_blocks {i} > logs/infer/infer-{i}.log 2>&1 &" # put it in the backend and run the next one concurrently
    logging.info(f"Running command: {command}")
    os.system(command)
