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

steps=[0,0,0,0,0,0,0,0,0,0,1]
worker_num = 6
# Loop over lora_layers from 0 to 31
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:2 or cpu)")
# parser.add_argument("--lora_layers", default=0,type=int, help="Number of lora blocks")
# parser.add_argument("--lora_dir", default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/without_head/trunk_full/ratio3/e12/{lora_layers}',type=str, help="Lora dir")
# parser.add_argument("--output_embedding_path", default='parameters/image/coco/val/embeddings_{v_block}_without_head.pth',type=str, help="embedding dir")


lora_dir='.checkpoints/lora/twitter/with_head/trunk/e50'
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
    for i in range(1,33):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_embedding_twitter_lora.py --lora_layers {i} --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # stop to run this command
        else:
            command = f"python get_embedding_twitter_lora.py --lora_layers {i}  --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[2]==1:
    for i in range(1,33):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_embedding_twitter_text.py --lora_layers {i} --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # stop to run this command
        else:
            command = f"python get_embedding_twitter_text.py --lora_layers {i}  --lora_dir {lora_dir} --embedding_dir {embeddings_path} --version {version}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[3]==1:
    worker_num = 6
    for i in range(1,33):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python test_twitter.py --vision_num_blocks {i} --version {version} --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1" # stop to run this command
        else:
            command = f"python test_twitter.py --vision_num_blocks {i} --version {version} --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1 &" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[4]==1:
    worker_num = 6
    for i in range(1,33):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python test_twitter_lora.py --vision_num_blocks {i}  --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1" # stop to run this command
        else:
            command = f"python test_twitter_lora.py --vision_num_blocks {i}  --lora_dir {lora_dir} --embeddings_path {embeddings_path}> logs/infer/infer-{i}.log 2>&1 &" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[5]==1:
    topk_list = [1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,150]
    worker_num = 12
    for i in topk_list:
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_layers_twitter.py --S {i} --version {version}" # stop to run this command
        else:
            command = f"python get_layers_twitter.py --S {i} --version {version}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[6]==1:
    topk_list = [1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,150]
    worker_num = 12
    for i in topk_list:
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_layers_twitter.py --S {i} --version {f'{version}-lora'}" # stop to run this command
        else:
            command = f"python get_layers_twitter.py --S {i} --version {f'{version}-lora'}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[7]==1:
    topk_list = [1,5, 10, 20, 30, 40]
    worker_num = 6
    for index, i in enumerate(topk_list):
        if index%(worker_num-1)==0 and index!=0:
            command = f"python model_predict_val.py --S {i} --version {version} --root {embeddings_path}> logs/twitter-val/predict_model-{i}.log 2>&1"
        else:
            command = f"python model_predict_val.py --S {i} --version {version} --root {embeddings_path}> logs/twitter-val/predict_model-{i}.log 2>&1 &"
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[8]==1:
    topk_list = [1,5, 10, 20, 30, 40]
    worker_num = 6
    for index, i in enumerate(topk_list):
        if index%(worker_num-1)==0 and index!=0:
            command = f"python model_predict_lora_val.py --S {i} --version {f'{version}-lora'} > logs/twitter-val/predict_model-{i}.log 2>&1"
        else:
            command = f"python model_predict_lora_val.py --S {i} --version {f'{version}-lora'} > logs/twitter-val/predict_model-{i}.log 2>&1 &"
        logging.info(f"Running command: {command}")
        os.system(command)

if steps[9]==1:
    Ns = [8]
    Qs = [1, 2, 5, 10, 20, 30, 40, 50, 60,70,80,90,100]
    Ss = [1,10,40]
    # Create a list of all combinations of Ns, Qs, and Ss
    combinations = [(N, Q, S) for S in Ss for N, Q in itertools.product(Ns, Qs)]

    # Number of GPUs
    num_gpus = 1

    # Generate the commands in a round-robin fashion across the GPUs
    commands = []
    for i, (N, Q, S) in enumerate(combinations):
        device = f"cuda:{i % num_gpus}"
        #version='val_model_v1' #with new lora predict models 
        #log_file = f'logs/ground_truth_val/e2e-val-S={S}_N={N}_Q={Q}_{version}.log'
        command = f"python e2e_twitter.py --N {N}  --Q {Q} --S {S}"
        commands.append(command)

    # Output the commands as a list of strings
    for command in commands:
        print(command)

    # Save commands to a file
    with open('commands.txt', 'w') as f:
        json.dump(commands, f)
        
    print("Commands have been saved to 'commands.txt'")

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

if steps[10]==1:
    Ns = [6,8,10]
    Qs = [1, 2, 5, 10, 20, 30, 40, 50, 60,70,80,90,100]
    Ss = [1,5,10,20,30,40]
    # Create a list of all combinations of Ns, Qs, and Ss
    combinations = [(N, Q, S) for S in Ss for N, Q in itertools.product(Ns, Qs)]

    # Number of GPUs
    num_gpus = 1

    # Generate the commands in a round-robin fashion across the GPUs
    commands = []
    for i, (N, Q, S) in enumerate(combinations):
        device = f"cuda:{i % num_gpus}"
        #version='val_model_v1' #with new lora predict models 
        #log_file = f'logs/ground_truth_val/e2e-val-S={S}_N={N}_Q={Q}_{version}.log'
        command = f"python e2e_twitter_lora.py --N {N}  --Q {Q} --S {S}"
        commands.append(command)

    # Output the commands as a list of strings
    for command in commands:
        print(command)

    # Save commands to a file
    with open('commands.txt', 'w') as f:
        json.dump(commands, f)
        
    print("Commands have been saved to 'commands.txt'")

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
