import subprocess
import torch
import numpy as np
import torch.nn.functional as F
# 遍历 audio_num_blocks 的值从 1 到 12 1,5 5,9  9,12    

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

Ns = [2, 6, 12] 
Qs = [5, 10, 20, 50, 100, 200, 500]
Ss = [1, 10, 30, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]
#for i in range(1,33):
for i in [1,10, 30, 50, 60,70,80,90,100,200,300,400,500,600]:
    # 构建执行命令
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    split='val'
    N=2
    K=100
    n=50
    model='model'
    # #get embeddings 
    # command = f"python image_trunks.py --lora_layers {i}  --split {split}"
    #command = f"python image_trunks_coco.py --lora_layers {i} "
    
    
    # #get maps 
    #for i in range(0,32):
    #command = f"python image_process.py --lora_layers {i}  --split 'train' "
    #command=f"python test_coco.py  {i}"
    
    
    # #get labels
    # #for i in [1,10, 30, 50, 60,70,80,90,100,200,300,400,500,600]:
    #command = f"python get_layers_image.py --S {i}  --split {split}"
    
    # #train models
    #for i in [1,10, 30, 50, 60,70,80,90,100,200,300,400,500,600]:
    # command = f"python image_predict_12.py --S {i}  --split {split}"
    
    #method3 to train models 
    # for n in range(1,33):
    #     command=f" python predict_model_3.py --S {i}  --split {split} --layer_num {n}"
    
    
    #e2e
    #for i in [1,10, 30, 50, 60,70,80,90,100,200,300,400,500,600]:
    command=f"python coco_mobile-search_engine_lora.py --S {i} --split {split} --N {N} --Q {Q} --device {device}"
    
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # parameter=one_image.total_params
    
    
    
    
    if result.returncode == 0:
        
        print(f"done")
    else:
        print(f"Command '{command}' failed with error: {result.stderr}")
    



    
