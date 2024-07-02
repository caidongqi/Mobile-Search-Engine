import logging
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
from api.coco_text2image import CoCo_t2i_Dataset
import argparse
from api.harsmart import HARsmart_relative

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--S", default=10,type=int, help="Number of S")
#split useless
parser.add_argument("--split", default='train',type=str, help="train or val")
parser.add_argument("--input_path", default='results/harsmart_lora_val_without_head',type=str, help="train or val")
parser.add_argument("--output_path", default='R{S}/layers.txt',type=str, help="train or val")


# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
split=args.split
input_path=args.input_path
output_path=args.output_path

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
    return 6

#file_path=f'/home/u2021010261/pc/Mobile-Search-Engine/results/imagenet/lora_{split}'

filenames = ['i{0}.txt'.format(i) for i in range(1,7)]
files = [os.path.join(f'{input_path}/R{S}',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]
matrix = np.vstack(counts)

global_batch_size = 256
test_ds = HARsmart_relative(train=False)
test_dl = DataLoader(dataset=test_ds, batch_size=global_batch_size, shuffle=False, drop_last=False,
num_workers=4, pin_memory=True, persistent_workers=True)

target_list=np.array([np.int64(i) for i  in range(len(test_ds))])

text_num=len(test_ds)

#sort matrix
sorted_column_indices,sorted_matrix=transform_matrix(target_list,matrix)


vision_layers=[]
vision_nums=[]
max_sum=-1
cur_image=0
for i,item in enumerate(sorted_column_indices):
    if (i==0):
        cur_item=item
    if (item==cur_item):
        rights=find_first_nonzero_row(sorted_matrix,i)
        if rights>max_sum:
            max_sum=rights
            max_sum_indices=i
        if i==len(sorted_column_indices)-1:
            vision_layers.append(max_sum_indices)   
            vision_nums.append(max_sum)
            
    else:
        cur_item=item
        vision_layers.append(max_sum_indices)   
        vision_nums.append(max_sum)
        max_sum=find_first_nonzero_row(sorted_matrix,i)
        max_sum_indices=i
        cur_image+=1
        if i==len(sorted_column_indices)-1:
            vision_layers.append(max_sum_indices)   
            vision_nums.append(max_sum)

#选取效果最好的text对应的数据
final_matrix=sorted_matrix[:,vision_layers]

for i in range(len(vision_layers)):
    if vision_nums[i]==33:
        vision_nums[i]=31

output_path=output_path.format(S=S)
np.savetxt(input_path+'/'+output_path,vision_nums,fmt='%d')
print(np.mean(vision_nums))
    
        