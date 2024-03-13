import logging
import torch
import data
import torchvision
import torchmetrics
import torch.nn as nn
from models import imagebind_model
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
logging.basicConfig(level=logging.INFO, force=True)
import os
csv_file_path = "/data/air/pc/Mobile-Search-Engine/datasets/clotho/clotho_captions_evaluation.csv"
data_dir="/data/air/pc/Mobile-Search-Engine/datasets/clotho/evaluation"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=["/data/air/pc/Mobile-Search-Engine/datasets/clotho/evaluation/"+file for file in audio_list]

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
#parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--audio_num_blocks", default=3,type=int, help="Number of audio blocks")
# 解析命令行参数
args = parser.parse_args()

# 获取 audio_num_blocks 的值
audio_num_blocks=args.audio_num_blocks

audio_num_blocks_1=audio_num_blocks
audio_num_blocks_2=12
device_ids = [1,2,3,4] 
device = "cuda:6" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_1 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_1)
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)
v_block=len(model_1.modality_trunks["vision"].blocks)
t_block=len(model_1.modality_trunks["text"].blocks)
a_block=len(model_1.modality_trunks["audio"].blocks)
i_block=len(model_1.modality_trunks["imu"].blocks)
model_1=model_1.cuda()
model_1 = model_1.to(device_ids[0]) 
model_1 = DataParallel(model_1,device_ids=device_ids)

model_1.eval()

# model_2.eval()
# model_2.to(device)
embeddings={}
import pandas as pd
with torch.no_grad():
    model_device = next(model_1.module.parameters()).device
    input={  
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=model_device)
    }
    embeddings[ModalityType.AUDIO] = model_1(input)[ModalityType.AUDIO]
    # 假设你想要保存模型参数和张量
    torch.save({
        'audio_embeddings': embeddings[ModalityType.AUDIO]
    }, f'embeddings_{a_block}.pth')