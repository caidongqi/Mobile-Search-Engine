import logging
import torch
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
csv_file_path = "/home/u2021010261/data/cdq/clotho/clotho_captions_evaluation.csv"
data_dir="/home/u2021010261/data/cdq/clotho/evaluation"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=["/data/air/pc/Mobile-Search-Engine/datasets/clotho/evaluation/"+file for file in audio_list]
embeddings={}


# # 1.存储n层 n=1
N=1
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_ids = [0,1,2] 
Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
batch_size=128
test_dl = DataLoader(dataset=Clotho_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

