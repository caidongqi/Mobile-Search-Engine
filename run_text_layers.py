import os
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
                    handlers=[logging.FileHandler(f'logs/run/get_embeddings{formatted_time}.log')], 
                    force=True)

steps=[1,0,1,0]
worker_num = 6
# Loop over lora_layers from 0 to 31
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:2 or cpu)")
# parser.add_argument("--lora_layers", default=0,type=int, help="Number of lora blocks")
# parser.add_argument("--lora_dir", default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/without_head/trunk_full/ratio3/e12/{lora_layers}',type=str, help="Lora dir")
# parser.add_argument("--output_embedding_path", default='parameters/image/coco/val/embeddings_{v_block}_without_head.pth',type=str, help="embedding dir")

head='with_head'
lora_dir='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/clotho/with_head/trunk/s1/e12'
embeddings_path=f'parameters/audio/clotho_val/with_head'
version='clotho_head'
dataset='clotho'
if steps[0]==1:
    for i in range(1,13):
        if i%(worker_num-1)==0 and i!=0:
            command = f"python get_embedding_clotho_text.py --lora_layers {i} --lora_dir {lora_dir} --embedding_dir {embeddings_path} --dataset {dataset}" # stop to run this command
        else:
            command = f"python get_embedding_clotho_text.py --lora_layers {i}  --lora_dir {lora_dir} --embedding_dir {embeddings_path} --dataset {dataset}" # put it in the backend and run the next one concurrently
        logging.info(f"Running command: {command}")
        os.system(command)