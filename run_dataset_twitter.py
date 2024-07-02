import os
import logging
import time
import itertools
import json
import subprocess
import threading
import queue

# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间
local_time = time.localtime(timestamp)
# 格式化本地时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'logs/run/get_embeddings{formatted_time}.log')], 
                    force=True)

steps=[1,0,0,0]
worker_num = 6
# Loop over lora_layers from 0 to 31
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:2 or cpu)")
# parser.add_argument("--lora_layers", default=0,type=int, help="Number of lora blocks")
# parser.add_argument("--lora_dir", default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/without_head/trunk_full/ratio3/e12/{lora_layers}',type=str, help="Lora dir")
# parser.add_argument("--output_embedding_path", default='parameters/image/coco/val/embeddings_{v_block}_without_head.pth',type=str, help="embedding dir")


lora_dir='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/clotho/with_head/trunk/s1/e12'
embeddings_path=f'parameters/image/twitter'
version='twitter'

if steps[0]==1:
    for i in range(1,33):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_embedding_twitter.py --lora_layers {i} --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # stop to run this command
        else:
            command = f"python get_embedding_twitter.py --lora_layers {i}  --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[1]==1:
    worker_num = 6
    for i in range(1,13):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python test_clotho_val.py --audio_num_blocks {i} --version {version} --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1" # stop to run this command
        else:
            command = f"python test_clotho_val.py --audio_num_blocks {i} --version {version} --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1 &" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[2]==1:
    topk_list = [1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,150,160,170,180,200,210,220,230,240,250,260,270,280,300,400,500,600]
    worker_num = 12
    for i in topk_list:
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_layers_clotho.py --S {i} --version {version}" # stop to run this command
        else:
            command = f"python get_layers_clotho.py --S {i} --version {version}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[3]==1:
    Ns = [7,8]
    Qs = [20,30]
    Ss = [200,300,400,500]

    # Create a list of all combinations of Ns, Qs, and Ss
    #combinations = [(N, S, Q) for Q in Qs for N, S in itertools.product(Ns, Ss)]

    combinations = [( S, Q) for Q in Qs for  S in  Ss]

    # Number of GPUs
    num_gpus = 1

    # Generate the commands in a round-robin fashion across the GPUs
    commands = []
    # for i, (N, S, Q) in enumerate(combinations):
    for i, ( S, Q) in enumerate(combinations):
        device = f"cuda:{i % num_gpus}"
        command = f"python e2e_clotho_val_ground_truth.py --S {S} --Q {Q} --version {version} --parameter_embedding_folder {embeddings_path}"
        commands.append(command)

    # Output the commands as a list of strings
    for command in commands:
        print(command)

    # Save commands to a file
    with open('commands.txt', 'w') as f:
        json.dump(commands, f)
        
    print("Commands have been saved to 'commands_true.txt'")

    num_workers = 6 # 一个gpu满载差不多跑6-7个worker，根据实际情况调整一下，不然会OOM

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
    with open('commands.txt', 'r') as f:
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
