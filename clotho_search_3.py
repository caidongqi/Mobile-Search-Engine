import logging
import torch
import data
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
data_dir="/home/u2021010261/data/cdq/clotho/evaluation/"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=[data_dir+file for file in audio_list]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_ids = [0,1,2] 
Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
batch_size=128
test_dl = DataLoader(dataset=Clotho_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

N=1
embedding_dynamic={}
with torch.no_grad():
        checkpoint = torch.load(f'parameters/dynamic/image/embeddings_{N}.pth')
        # 获取模型参数和张量
        embedding_dynamic[ModalityType.AUDIO]= checkpoint['audio_embeddings']
        print(1)

# #3 根据query进行match到前k个数据
fine_model = imagebind_model.imagebind_huge(pretrained=True)
fine_model=fine_model.to(device)
top_k={}
topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600,700,800,900,1000]
counts_rs = {}
shortlist={}
shortlist_item={}
for k in topk1:
        counts_rs[f'counts_r{k}'] = np.array([])
        shortlist[f'counts_r{k}']=[]
        shortlist_item[f'counts_r{k}']=[]
    
K=10
counts_r1=np.array([])
counts_r10=np.array([])
for batch_idx, (x, target) in enumerate(test_dl):
        target = target.to(device)
        inputs = {
        ModalityType.TEXT: data.load_and_transform_text(x, device)
        }
        fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.AUDIO].device)
        match_value_1 = fine_embeddings @ embedding_dynamic[ModalityType.AUDIO].T 
        result_1 = torch.softmax(match_value_1, dim=-1)
        _, predicted = torch.max(result_1, dim=-1)
        _, topk_indices = torch.topk(result_1, k=K, dim=-1)
        counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
            # #counts_r1 = np.concatenate([counts_r1, [any(predicted[i] == target[i]) for i in range(len(predicted))]])
            # #topk_indices=topk_indices.T
        counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
        
        top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]
        
        for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                        counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                        
                else:
                        counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                        # for i,row in enumerate(top_indices):
                        #     list=[]
                        #     list_item=[]
                        #     for item in row:
                        #         list.append(audio_path[item])
                        #         list_item.append(item.item())
                        #     shortlist[counts_r].append(list)
                        #     shortlist_item[counts_r].append(list_item)

                for i,row in enumerate(top_indices):
                    list=[]
                    list_item=[]
                    for item in row:
                        list.append(audio_path[item])
                        list_item.append(item.item())
                    shortlist[counts_r].append(list)
                    shortlist_item[counts_r].append(list_item)
            
import pickle

# 假设你有一个文件路径用于保存数据
file_path = "shortlist_data.pkl"

# 保存 shortlist 和 shortlist_item 到本地文件
with open(file_path, 'wb') as f:
    pickle.dump(shortlist, f)
    pickle.dump(shortlist_item, f)

print("Data saved successfully to", file_path)
import pickle

# 文件路径
file_path = "shortlist_data.pkl"

# 提取数据
with open(file_path, 'rb') as f:
    shortlist = pickle.load(f)
    shortlist_item = pickle.load(f)

# 打印结果
print("Shortlist:", shortlist)
print("Shortlist Item:", shortlist_item)