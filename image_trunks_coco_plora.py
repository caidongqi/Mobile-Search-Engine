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
from api.clotho import ClothoDataset
from api.coco import CoCoDataset
logging.basicConfig(level=logging.INFO, force=True)
import os
import itertools
from api.imagenet import ImageNetDataset
imagenet_datadir = "/home/u2021010261/pc/imagenet/imagenet"

import argparse
# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--lora_layers", default=0,type=int, help="Number of audio blocks")

# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
lora_layers=args.lora_layers

device = "cuda:0" if torch.cuda.is_available() else "cpu"
lora=True
linear_probing=False
load_head_post_proc_finetuned=True
lora_dir=f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/trunk/ratio3/e12/{lora_layers+1}'
#lora_dir =f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/imagenet/step1/{vision_num_blocks}'

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_1 = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=lora_layers+1)
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)
v_block=len(model_1.modality_trunks["vision"].blocks)
t_block=len(model_1.modality_trunks["text"].blocks)
a_block=len(model_1.modality_trunks["audio"].blocks)
i_block=len(model_1.modality_trunks["imu"].blocks)

if lora:
    model_1.modality_trunks.update(LoRA.apply_lora_modality_trunks(model_1.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.VISION: [i for i in range(1,lora_layers+1)]},
                                        modality_names=[ModalityType.VISION]))
 
    LoRA.load_lora_modality_trunks(model_1.modality_trunks, checkpoint_dir=lora_dir, postfix = "_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model_1.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir, device = device)
        load_module(model_1.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir, device = device)
elif linear_probing:
    # Load heads
    load_module(model_1.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir, device = device)
 
model_1=model_1.cuda()
model_1 = model_1.to(device) 
model_1.eval()

# model_2.eval()
# model_2.to(device)
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
   
    coco_datadir="/home/u2021010261/share/pc/COCO/val2017"
    coco_annotation_file='/home/u2021010261/share/pc/COCO/instances_val2017.json'
    test_ds=CoCoDataset(datadir=coco_datadir, annFile=coco_annotation_file,transform=data_transform)
    test_dl=DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
   
    
    for batch_idx, (x, target) in enumerate(test_dl):
            #model_device = next(model_1.module.parameters()).device        
            x = x.to(device)
            target=[t.to(device) for t in target]
            inputs = {
                ModalityType.VISION: x,
                           }
           # embeddings = model_1(inputs)
            current_embeddings = model_1(inputs)[ModalityType.VISION]
            
            if embeddings:
                embeddings[ModalityType.VISION] = torch.cat([embeddings[ModalityType.VISION], current_embeddings], dim=0)
            else:
                embeddings[ModalityType.VISION] = current_embeddings
    
            # 释放之前的中间结果
            del current_embeddings
            
            #embeddings_list.append(model_1(inputs)[ModalityType.VISION])
            #embeddings[ModalityType.VISION]=torch.cat((embeddings[ModalityType.VISION],model_1(inputs)[ModalityType.VISION]),dim=1)
    # 假设你想要保存模型参数和张量
    # 使用 torch.cat() 将列表中的张量连接起来
    embeddings[ModalityType.VISION] = embeddings[ModalityType.VISION].to(device)
    #embeddings[ModalityType.VISION] = torch.cat(embeddings_list, dim=1)
    torch.save({
        'vision_embeddings': embeddings[ModalityType.VISION]
    }, f'parameters/image/coco/val/embeddings_{v_block}_plora.pth')

    