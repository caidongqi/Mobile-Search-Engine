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
                    handlers=[logging.FileHandler(f'logs/run/labels_{formatted_time}.log')], 
                    force=True)

topk_list = [1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]

# for index, i in enumerate(topk_list):
#     command = f"python get_layers_image.py --S {i}"
#     logging.info(f"Running command: {command}")
#     os.system(command)

worker_num = 12
for i in topk_list:
    if i%(worker_num-1)==0 and i!=0:
        command = f"python get_layers_flickr8k.py --S {i} " # stop to run this command
    else:
        command = f"python get_layers_flickr8k.py --S {i}" # put it in the backend and run the next one concurrently
    logging.info(f"Running command: {command}")
    os.system(command)
