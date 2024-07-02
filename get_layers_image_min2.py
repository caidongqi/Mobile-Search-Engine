import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
from api.coco_text2image import CoCo_t2i_Dataset
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--S", default=10,type=int, help="Number of S")
parser.add_argument("--split", default='train',type=str, help="train or val")

# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
split=args.split

def transform_matrix(indices,matrix):
    if len(indices)!=matrix.shape[1]:
        raise ValueError("Length of column_indices must match the number of columns in the matrix.")

    sorted_indices=np.argsort(indices)
    sorted_matrix=matrix[:,sorted_indices]
    sorted_sequence=indices[sorted_indices]
    return sorted_sequence,sorted_matrix
  

def find_first_nonzero_row(array, column_index):
    for i, row in enumerate(array):
        if row[column_index] != 0:
            return i
    return 33

def get_max_layer(indices,matrix):
    vision_layers=[]
    vision_nums=[]
    max_sum=-1
    cur_image=0
    for i,item in enumerate(indices):
        if (i==0):
            cur_item=item
        if (item==cur_item):
            rights=find_first_nonzero_row(matrix,i)
            if rights>max_sum:
                max_sum=rights
                max_sum_indices=i
            if i==len(indices)-1:
                vision_layers.append(max_sum_indices)   
                vision_nums.append(max_sum)
        else:
            cur_item=item
            vision_layers.append(max_sum_indices)   
            vision_nums.append(max_sum)
            max_sum=find_first_nonzero_row(matrix,i)
            max_sum_indices=i
            cur_image+=1

    return vision_nums,vision_layers

def get_min_layer(indices,matrix):
    vision_layers=[]
    vision_nums=[]
    min_sum=33
    cur_image=0
    for i,item in enumerate(indices):
        if (i==0):
            cur_item=item
        if (item==cur_item):
            rights=find_first_nonzero_row(matrix,i)
            if rights<min_sum:
                min_sum=rights
                min_sum_indices=i
            if i==len(indices)-1:
                vision_layers.append(min_sum_indices)   
                vision_nums.append(min_sum)
        else:
            cur_item=item
            vision_layers.append(min_sum_indices)   
            vision_nums.append(min_sum)
            min_sum=find_first_nonzero_row(matrix,i)
            min_sum_indices=i
            cur_image+=1

    return vision_nums,vision_layers

#file_path=f'/home/u2021010261/pc/Mobile-Search-Engine/results/imagenet/lora_{split}'
file_path=f'results/coco_lora_val_nohead'
filenames = ['v{0}_t24.txt'.format(i) for i in range(1,33)]
files = [os.path.join(f'{file_path}/R{S}',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]
matrix = np.vstack(counts)

CoCo_dataset = CoCo_t2i_Dataset(split="val",images_dir='/home/share/pc/COCO/val2017',caption_path='/home/share/pc/COCO/captions_val2017.json')
test_dl = DataLoader(dataset=CoCo_dataset, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
target_list=[]
#get target order
for batch_idx, (x, target) in enumerate(test_dl):
    target_list=np.append(target_list,np.array(target))

target_list=target_list.astype(int)

text_num=len(CoCo_dataset)

#sort matrix
sorted_column_indices,sorted_matrix= transform_matrix(target_list,matrix)

#选取效果最好的text对应的数据
vision_nums,vision_layers=get_min_layer(sorted_column_indices,sorted_matrix)

final_matrix=sorted_matrix[:,vision_layers]

for i in range(len(vision_layers)):
    if vision_nums[i]==33:
        vision_nums[i]=31
np.savetxt(f'{file_path}/R{S}/layers_min.txt',vision_nums,fmt='%d')
print(np.mean(vision_nums))
    
        