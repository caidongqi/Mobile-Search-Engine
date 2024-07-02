import subprocess
import json
import threading
import queue

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
                    handlers=[logging.FileHandler(f'logs/run/e2e_{formatted_time}.log')], 
                    force=True)

# Number of concurrent workers
num_workers = 6 # 一个gpu满载差不多跑6-7个worker，根据实际情况调整一下，不然会OOM

# import time
# logging.info("Starting to sleep 20mins for waiting the end of embedding extraction")
# time.sleep(1200)
# logging.info("Finished sleeping")

# Clean
# rm -rf logs/coco/*
# rm -rf parameters/image/coco/shortlist/*


# Function to execute a command
def run_command(cmd_queue):
    while not cmd_queue.empty():
        command = cmd_queue.get()
        logging.info(f"Executing: {command}")
        result = subprocess.run(command, shell=True)
        if result.returncode == 0:
            logging.info("Command completed successfully.")
        else:
            logging.info(f"Command failed with error: {result.stderr}")
        cmd_queue.task_done()

# Read commands from the file
with open('commands_flicker.txt', 'r') as f:
    commands = json.load(f)

# Create a queue and add commands to it
cmd_queue = queue.Queue()
for command in commands:
    cmd_queue.put(command)



# Create and start threads
threads = []
for _ in range(num_workers):
    t = threading.Thread(target=run_command, args=(cmd_queue,))
    t.start()
    threads.append(t)

# Wait for all threads to complete
for t in threads:
    t.join()

logging.info("All commands have been executed.")
