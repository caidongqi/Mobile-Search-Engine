import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import csv
import torch
import torch.nn as nn
import torchaudio
import json
class CoCo_t2i_Dataset(Dataset):
    def __init__(self, json_file,device='cpu',datadir=''):
        # 初始化操作，可以在这里加载数据集
        self.dir=datadir
        dataset = os.listdir(datadir)
        self.datadir=sorted(dataset)
        with open(json_file, 'r') as json_file:
            self.data= json.load(json_file)
        img_id=[]
        self.annotations=self.data['annotations']
        
        img_maps={}
        for i,item in enumerate(self.datadir):
            img_maps[item]=i
        self.img_maps=img_maps
        

        text_maps={}
        text_list=[]
        for item in self.annotations:
            text_maps[item['caption']]=item['image_id']
            text_list.append(item['caption'])
        self.text_list=text_list
        self.text_maps=text_maps
        self.device=device

    def __len__(self):
        # 返回数据集的长度
        file_list = os.listdir(self.dir)
        
        # 使用列表推导式过滤出所有文件，而不包括子文件夹  
        return len(file_list)
    

    def __getitem__(self, idx):
        text=self.text_list[idx]
        id=self.text_maps[text]
        formatted_number = str(id).zfill(12)  # 填充到12位
        filename = formatted_number + '.jpg'
        target=self.img_maps[filename]
       
        return text,target
        # 在这里可以进行数据转换操作，如果定义了 transform

        

# # # # 使用示例
# csv_file_path = "/home/u2021010261/share/pc/COCO/captions_val2017.json"
# data_dir="/home/u2021010261/share/pc/COCO/val2017"
# device="cuda:0"
# CoCo_dataset = CoCo_t2i_Dataset(json_file=csv_file_path,datadir=data_dir,device=device)
# test_dl = DataLoader(dataset=CoCo_dataset, batch_size=64, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)

# with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(test_dl):
#             print(x)
#             print(target)
#             x=x.to(device)
#             target = target.to(device)