import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import csv
import torch
import torch.nn as nn
import torchaudio
class ClothoDataset(Dataset):
    def __init__(self, csv_file,device='cpu',datadir=''):
        # 初始化操作，可以在这里加载数据集
        self.dir=datadir
        dataset = os.listdir(datadir)
        self.datadir=sorted(dataset)
        self.data = pd.read_csv(csv_file,sep=',') # 假设数据集以CSV文件形式提供
        captions_array = self.data[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()

# 打印数组
        #print(captions_array)
        self.text_list=captions_array
        self.device=device

    def __len__(self):
        # 返回数据集的长度
        file_list = os.listdir(self.dir)
        
        # 使用列表推导式过滤出所有文件，而不包括子文件夹  
        return len(file_list)
    

    def __getitem__(self, idx):
        # 根据索引获取单个样本
        # sample = self.data.iloc[idx]
        # audio=self.data
        #dir_path=self.datadir[idx]
        dir=self.datadir[idx]
        dir_path = os.path.join(self.dir, self.datadir[idx])
        for row_number, row in self.data.iterrows():
            # 检查文件名是否在当前行的第一列
            # names_id=dir[:11]
            # names_num=dir[12:18]
            if row['file_name'].startswith(dir):
                # 返回找到的行数
                      #label = row['class']
                      captions = self.data.iloc[row_number, 1:].values.tolist()
                      index=[self.text_list.index(item) for item in captions]
                      index=torch.tensor(index)
                      #index = self.text_list.index(label)
                      return dir_path,index
        
        
        # 在这里可以进行数据转换操作，如果定义了 transform

        

# # # 使用示例
# csv_file_path = "/home/pc/Mobile-Search-Engine/datasets/clotho_captions_evaluation.csv"
# data_dir="/home/pc/Mobile-Search-Engine/datasets/evaluation"
# device="cuda:0"
# Clotho_dataset = ClothoDataset(csv_file=csv_file_path,datadir=data_dir,device=device)
# test_dl = DataLoader(dataset=Clotho_dataset, batch_size=64, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)

# with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(test_dl):
#             #x=x.to(device)
#             target = target.to(device)