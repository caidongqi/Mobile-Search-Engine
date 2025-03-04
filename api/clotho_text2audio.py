import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import csv
import torch
import torch.nn as nn
import torchaudio
import math
class ClothoTextDataset(Dataset):
    def __init__(self, csv_file, device='cpu'):
        # 初始化操作，可以在这里加载数据集
        self.data = pd.read_csv(csv_file,sep=',') # 假设数据集以CSV文件形式提供
        captions_array = self.data[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()

# 打印数组
        #print(captions_array)
        self.text_list=captions_array
        self.device=device

    def __len__(self):
        # 返回数据集的长度
        # file_list = os.listdir(len(self.text_list))
        
        # 使用列表推导式过滤出所有文件，而不包括子文件夹  
        return len(self.text_list)
    

    def __getitem__(self, idx):
        # 根据索引获取单个样本
        # sample = self.data.iloc[idx]
        # audio=self.data
        #dir_path=self.datadir[idx]
        text=self.text_list[idx]
        #dir_path = os.path.join(self.dir, self.datadir[idx])
        id=math.floor(idx/5)
        return text,id
        
        
        # 在这里可以进行数据转换操作，如果定义了 transform

        

# # # 使用示例
# csv_file_path = "/home/pc/Mobile-Search-Engine/datasets/clotho_captions_evaluation2.csv"
# #data_dir="/home/pc/Mobile-Search-Engine/datasets/evaluation"
# data = pd.read_csv(csv_file_path)  # 替换为实际的CSV文件路径
# # 打印数据前几行
# print("Data head:", data.head())

# device="cuda:0"
# Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
# test_dl = DataLoader(dataset=Clotho_dataset, batch_size=64, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)
# with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(test_dl):
#             #x=x.to(device)
#             target = target.to(device)