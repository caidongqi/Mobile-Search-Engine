import logging
import torch
import json
import data
import torchvision
import torchmetrics
import pickle
import math
import itertools
from models import imagebind_model,mlp
from models.mlp import MyModel
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from api.imagenet import ImageNetDataset
from api.clotho import ClothoDataset
logging.basicConfig(level=logging.INFO, force=True)
import os
N=3
K=1000

imagenet_datadir = "/home/u2021010261/data/yx/imagenet"
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
test_ds1 = ImageNetDataset(datadir=imagenet_datadir, split="val", transform=data_transform)
    #test_dl1=DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_dl1 = DataLoader(dataset=test_ds1, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
num_samples = 5000
test_dl = DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

test_dl = itertools.islice(test_dl, num_samples)
    
# csv_file_path = "/home/u2021010261/data/cdq/clotho/clotho_captions_evaluation.csv"
# data_dir="/home/u2021010261/data/cdq/clotho/evaluation"
# f_s=os.listdir(data_dir)
# print(len(f_s))
# pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
# text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
# audio_list=pf[['file_name']].values.flatten().tolist()
# audio_path=["/home/u2021010261/data/cdq/clotho/evaluation/"+file for file in audio_list]
embeddings={}
device = "cuda:1" if torch.cuda.is_available() else "cpu"
device_ids = [0,1,3,4,6,7] 
# Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
# batch_size=128
# test_dl = DataLoader(dataset=Clotho_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)




# # 1.存储n层 n=1
# N=1
embedding_folder=f'parameters/image/trunks+post/embeddings_{N}.pth'
if os.path.exists(embedding_folder):
    with torch.no_grad():
            checkpoint = torch.load(embedding_folder)
            # 获取模型参数和张量
            embeddings[ModalityType.VISION]= checkpoint['audio_embeddings']
            print(1)

else :
    print(1)#process audio/image 这里要加一个保存embedding的函数




# 2.每个数据动态存储 m 层
# 创建模型实例
input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 12  # 根据层数的范围确定输出层大小
predict_model = MyModel(input_size, output_size)  # 请确保input_size和output_size已定义
predict_model.to(device)
# 加载已保存的模型参数
predict_model.load_state_dict(torch.load('/home/u2021010261/pc/Mobile-Search-Engine/parameters/model/model_trunks12_parameters.pth'))
models = []

layers_folder=f"parameters/layers/imagenet/layers_{N}.pkl"
if os.path.exists(layers_folder):
    # 打开数据文件
    with open(layers_folder, 'rb') as f:
        layers = pickle.load(f)
else:
    layers=[]
    for embedding_item in embeddings[ModalityType.AUDIO]:
            embedding_item=embedding_item.to(device)
            layer=predict_model(embedding_item.float())
            _, layer1 = torch.max(layer, 0)
            layers.append(layer1+1)
    # 保存数据到文件
    with open(layers_folder, 'wb') as f:
        pickle.dump(layers, f)
        
embedding_dynamic_folder=f'parameters/dynamic/image/embeddings_{N}.pth'
if os.path.exists(embedding_dynamic_folder):
    embedding_dynamic={}
    with torch.no_grad():
            checkpoint = torch.load(embedding_dynamic_folder)
            # 获取模型参数和张量
            embedding_dynamic[ModalityType.VISION]= checkpoint['audio_embeddings']
            print(1)
else:
    embedding_dynamic={}
    with torch.no_grad():
        for i in range(len(layers)):
                parameter_embedding_folder=f'parameters/image/trunks+post/embeddings_{layers[i]}.pth'
                if os.path.exists(parameter_embedding_folder):
                    current_embeddings = torch.load(parameter_embedding_folder, map_location=torch.device('cuda'))['audio_embeddings'][i]
                    
                    #current_embeddings=torch.load(parameter_embedding_folder)['audio_embeddings'][i]
                    if embedding_dynamic:
                            embedding_dynamic[ModalityType.AUDIO] = torch.cat([embedding_dynamic[ModalityType.AUDIO], current_embeddings.unsqueeze(0).to(embedding_dynamic[ModalityType.AUDIO].device)], dim=0)
                    else:
                            embedding_dynamic[ModalityType.AUDIO] = current_embeddings.unsqueeze(0)
                    del current_embeddings
                    print(embedding_dynamic[ModalityType.AUDIO].shape)
                else:
                    print("no parameters/image/trunks+post/embeddings_{layers[i]}.pth ")
                    # inputs = {
                    # ModalityType.AUDIO: data.load_and_transform_audio_data2(audio_path[i],device=device_ids[math.ceil((layers[i]-1)/2)])
                    # }
                    
                    # current_embeddings = models[layers[i]](inputs)[ModalityType.AUDIO]

                    # if embedding_dynamic:
                    #         embedding_dynamic[ModalityType.AUDIO] = torch.cat([embedding_dynamic[ModalityType.AUDIO], current_embeddings.unsqueeze(0).to(embedding_dynamic[ModalityType.AUDIO].device)], dim=0)
                    # else:
                    #         embedding_dynamic[ModalityType.AUDIO] = current_embeddings.unsqueeze(0)

                    # del current_embeddings
        torch.save({
                'audio_embeddings': embedding_dynamic[ModalityType.AUDIO]
            }, embedding_dynamic_folder)






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
    
counts_r1=np.array([])
counts_r10=np.array([])
shortlist_folder=f"parameters/shortlist/imagenet/shortlist_data_{N}.pkl"
if os.path.exists(shortlist_folder):
    with open(shortlist_folder, 'rb') as f:
        shortlist = pickle.load(f)
        shortlist_item = pickle.load(f)
    
else:    
    for batch_idx, (x, target,imgs) in enumerate(test_dl):
            target = target.to(device)
            inputs = {
            ModalityType.TEXT:  data.load_and_transform_text(test_ds1.text_list, device)
            }
            #fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.AUDIO].device)
            fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.VISION].device)
            match_value_1 = embedding_dynamic[ModalityType.VISION][batch_idx]@fine_embeddings.T
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
                            list.append(test_ds1.text_list[item])
                            list_item.append(item.item())
                        #shortlist[counts_r].append(list)
                        shortlist_item[counts_r].append(list_item)
            
            # 假设你有一个文件路径用于保存数据
            file_path=os.path.dirname(shortlist_folder)
            # 保存 shortlist 和 shortlist_item 到本地文件
            with open(file_path, 'wb') as f:
                #pickle.dump(shortlist, f)
                pickle.dump(shortlist_item, f)

            print("Data saved successfully to", file_path)



# 4 再次进行fine-grained embedding
# test_dl2 = DataLoader(dataset=Clotho_dataset, batch_size=1, shuffle=False, drop_last=False,
#              num_workers=4, pin_memory=True, persistent_workers=True)
batch_size=1        
counts_r1=np.array([])
counts_r10=np.array([])
topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600,700,800,900,1000]
counts_rs = {}
for k in topk1:
    if k<=K:
        counts_rs[f'counts_r{k}'] = np.array([])
embeddings={}
embeddings_all={}

with torch.no_grad():
        checkpoint = torch.load(f'parameters/image/trunks+post/embeddings_32.pth')
        # 获取模型参数和张量
        embeddings_all[ModalityType.TEXT]= checkpoint['audio_embeddings']
        print(1)

        for batch_idx, (x, target,imgs) in enumerate(test_dl):
            embeddings_TEXT={}
            text_list=[]
            target = target.to(device)
            for item in shortlist_item[f'counts_r{K}'][batch_idx*batch_size]:
                text_list.append(test_ds1.text_list[item])
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list, device)
                #ModalityType.AUDIO: data.load_and_transform_audio_data(shortlist[f'counts_r{K}'][batch_idx*batch_size],device=device)
            }
            
        
            
            #fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.AUDIO].device)
            fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.VISION].device)
            match_value_1 = embedding_dynamic[ModalityType.VISION][batch_idx]@fine_embeddings.T
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, dim=-1)
            _, topk_indices = torch.topk(result_1, k=K, dim=-1)
            counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                # #counts_r1 = np.concatenate([counts_r1, [any(predicted[i] == target[i]) for i in range(len(predicted))]])
                # #topk_indices=topk_indices.T
            counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            
            
    
            top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] if k <= K else None for k in topk1]
            r1=0
            r10=0
            predicted=torch.Tensor([shortlist_item[f'counts_r{K}'][batch_idx*batch_size][predicted]])
        
            #_, topk_indices = torch.topk(result_1, k=10, dim=-1)
            #top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] if k <= len(shortlist_item[f'counts_r{K}'][batch_idx]) else None for k in topk1]
            for k, top_indices, counts_rk in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                        counts_rs[counts_rk] = np.concatenate([counts_rs[counts_rk], [int(predicted[i] == target.to(predicted.device)) for i in range(len(predicted))]])
                elif k<=K:            
                        results_k=[]
                        for i_k in range(k):
                            results_k.append(shortlist_item[f'counts_r{K}'][batch_idx][top_indices[0][i_k].item()])
                        counts_rs[counts_rk] = np.concatenate([counts_rs[counts_rk], [int(any(results_k[i] == target.to(predicted.device))) for i in range(len(results_k))]])
                        
                else :
                    break
            r1=(np.sum(counts_rs['counts_r1']))/len(counts_rs['counts_r1'])
            r10=(np.sum(counts_rs['counts_r10']))/len(counts_rs['counts_r1']) 
        
            logging.info(f"batch_idx = {batch_idx}, r1={r1},r10={r10}, test_total = {len(counts_r1)}")
    
results=[]
lists=[]
for counts in counts_rs:
    correct=np.sum(counts_rs[counts] == 1)/len(counts_rs['counts_r1'])
    results.append(str(correct))
    lists.append(str(counts))
lists.append(['N'])
lists.append(['K'])
results.append(K)
results.append(N)
import csv

# 数据
data1 = [
    results
]

# # 指定CSV文件路径
csv_file_path = 'end_to_end.csv'

# # 将数据写入CSV文件
# with open(csv_file_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
    
#     # 写入数据
#     for row in data1:
#         writer.writerow(row)

# print("CSV文件已保存至:", csv_file_path)
with open(csv_file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入新数据
    for row in data1:
        writer.writerow(row)