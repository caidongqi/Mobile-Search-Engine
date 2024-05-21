import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO, force=True)
import os
import random
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--S", default=10,type=int, help="Number of S")
parser.add_argument("--split", default='train',type=str, help="train or val")

# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
split=args.split
def find_first_nonzero_row(array, column_index):
    for i, row in enumerate(array):
        if row[column_index] != 0:
            return i
    return 33

#file_path=f'/home/u2021010261/pc/Mobile-Search-Engine/results/imagenet/lora_{split}'
file_path=f'/home/u2021010261/share/pc/Mobile-Search-Engine/results/coco'
filenames = ['v{0}_t24.txt'.format(i) for i in range(1,33)]
files = [os.path.join(f'{file_path}/R{S}',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]
vision_layers={}
for i in range(len(counts[0])):
    vision_layers[i]=[]
for i in range(len(counts[0])):
    vision_id=i    
    vision_layers[i].append(find_first_nonzero_row(counts,i))
vision_max=[]
for i in range(len(counts[0])):
    vision_id_id=i
    vision_max.append(min(vision_layers[i]))
for i in range(len(counts[0])):
    if vision_max[i]==33:
        vision_max[i]=0
np.savetxt(f'{file_path}/R{S}/layers.txt',vision_max,fmt='%d')
    
        