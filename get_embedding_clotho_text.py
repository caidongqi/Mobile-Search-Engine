import logging
import torch
import data
import torchvision
import torchmetrics
from torch.utils.data import Subset
import torch.nn as nn
from models import imagebind_model
from models import imagebind_model_only_trunks
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from api.clotho_text2audio import ClothoTextDataset
logging.basicConfig(level=logging.INFO, force=True)
import os
import itertools
from api.imagenet import ImageNetDataset
imagenet_datadir = "/home/u2021010261/pc/imagenet/imagenet"

import argparse
from api.flickr import flickr8k
# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--lora_layers", default=0,type=int, help="Number of audio blocks")
parser.add_argument("--lora_dir", default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/clotho/with_head/trunk/s1/e12',type=str, help="Number of audio blocks")
parser.add_argument("--embedding_dir", default='parameters/audio/clotho_val/with_head',type=str, help="Number of audio blocks")
parser.add_argument("--dataset", default='clotho_val',type=str, help="Number of audio blocks")


# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
lora_layers=args.lora_layers
lora_dir=args.lora_dir
embedding_dir=args.embedding_dir
dataset=args.dataset
version='head'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
lora=True
linear_probing=False
load_head_post_proc_finetuned=True
lora_dir=f'{lora_dir}/{lora_layers}'
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
                    handlers=[logging.FileHandler(f'logs/embeddings/{dataset}_{lora_layers}_{version}_{formatted_time}.log')], 
                    force=True)
#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=lora_layers,vision_num_blocks=0)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)
embedding_dir=f'{embedding_dir}/text_embeddings_{a_block}.pth'


if lora:
    model.modality_trunks.update(LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.AUDIO: [i for i in range(0,lora_layers)]},
                                        modality_names=[ModalityType.AUDIO]))
 
    LoRA.load_lora_modality_trunks(model.modality_trunks, checkpoint_dir=lora_dir, postfix = "_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir, device = device)
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir, device = device)
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir, device = device)
 
model=model.cuda()
model = model.to(device) 
model.eval()

embeddings={}
embeddings_list=[]

import pandas as pd
with torch.no_grad():
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
   
    csv_file='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.datasets/data/clotho_csv_files/clotho_captions_evaluation.csv'
    Clotho_dataset = ClothoTextDataset(csv_file=csv_file, device=device)
    test_dl = DataLoader(dataset=Clotho_dataset, batch_size=64, shuffle=False, drop_last=False,
                num_workers=4, pin_memory=True, persistent_workers=True)
    
    for batch_idx, (x, target) in enumerate(test_dl):
        target = target.to(device)
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(x, device)   
        }
        
        # embeddings = model(inputs)
        current_embeddings = model(inputs)[ModalityType.TEXT]
        
        if embeddings:
            embeddings[ModalityType.TEXT] = torch.cat([embeddings[ModalityType.TEXT], current_embeddings], dim=0)
        else:
            embeddings[ModalityType.TEXT] = current_embeddings

        # 释放之前的中间结果
        del current_embeddings
        logging.info(f"batch_idx: {batch_idx}")
    
            
    embeddings[ModalityType.TEXT] = embeddings[ModalityType.TEXT].to(device)
    logging.info(f"embeddings computed")
    
    torch.save({
        'text_embeddings': embeddings[ModalityType.TEXT]
    }, embedding_dir)

    