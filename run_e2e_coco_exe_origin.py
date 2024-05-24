import subprocess
import json
import threading
import queue

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

import time
logging.info("Starting to sleep 30mins for waiting the end of embedding extraction")
time.sleep(1800)
logging.info("Finished sleeping")

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
with open('commands_origin.txt', 'r') as f:
    commands = json.load(f)

# Create a queue and add commands to it
cmd_queue = queue.Queue()
for command in commands:
    cmd_queue.put(command)

# Number of concurrent workers
num_workers = 32

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
