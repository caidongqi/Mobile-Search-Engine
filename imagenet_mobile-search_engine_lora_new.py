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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)
import os

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--N", type=int, default=2, help="First get N embeddings")
parser.add_argument("--K", default=100,type=int, help="Number of K")
parser.add_argument("--S", default=60,type=int, help="Number of S")
parser.add_argument("--split", default='train',type=str, help="train or val")
parser.add_argument("--device", default='cuda:0',type=str, help="gpu device id (if applicable)")

args = parser.parse_args()
N=args.N
K=args.K
S=args.S
split=args.split
version=1
# N=2
# K=100
logging.info(f"N={N},K={K}")
full_layer=32 #audio:12 image:32
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
# test_ds1 = ImageNetDataset(datadir=imagenet_datadir, split="val", transform=data_transform)
#     #test_dl1=DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
# test_dl1 = DataLoader(dataset=test_ds1, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
# num_samples = 5000
# batch_size=64
# test_dl = DataLoader(dataset=test_ds1, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
# test_dl = itertools.islice(test_dl, num_samples)
test_ds1 = ImageNetDataset(datadir=imagenet_datadir, split='val', transform=data_transform)
    #test_dl1=DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_dl1 = DataLoader(dataset=test_ds1, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
num_samples = 2500
batch_size=32
test_dl = DataLoader(dataset=test_ds1, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
test_dl = itertools.islice(test_dl, num_samples)


test_dl_subset = DataLoader(dataset=test_ds1, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
test_dl_subset = itertools.islice(test_dl_subset, num_samples)
# 将迭代器转换为列表
subset_data = list(test_dl_subset)
# 创建新的 DataLoader
test_dl2 = DataLoader(dataset=subset_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)

test_dl2=test_dl
#embedding path
parameter_embedding_folder=f'parameters/image/lora/trunks/' # e2e still use val
lora_dir =f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/imagenet/step1/31'
model_parameter=f'parameters/model/imagenet_{split}/image_S={S}.pth'
#model_parameter=f"parameters/image/lora/trunks_{split}/model/method3/image_{split}_R{S}_layer={layer_num}.pth"
#parameters/model/imagenet_{split}/image_S={S}.pth


embedding_folder=f'{parameter_embedding_folder}embeddings_{N}.pth'
embedding_dynamic_folder=f'{parameter_embedding_folder}dynamic/N={N}_S={S}_v{version}.pth'
#save layers
layers_folder=f"{parameter_embedding_folder}layers/N={N}_S={S}_v{version}.pkl"


shortlist_folder=f"{parameter_embedding_folder}shortlist/shortlist_data_{N}_{K}_S={S}_v{version}.pkl"

fine_model_embeddings=f'{parameter_embedding_folder}embeddings_{full_layer}.pth'
embeddings={}
device = args.device if torch.cuda.is_available() else "cpu"


# 1.存储n层 n=1
if os.path.exists(embedding_folder):
    with torch.no_grad():
            checkpoint = torch.load(embedding_folder)
            # 获取模型参数和张量
            embeddings[ModalityType.VISION]= checkpoint['audio_embeddings']
            
            logging.info('步骤1已加载')

else :
    
    logging.info('步骤1未加载')
    logging.info(f'没有参数{embedding_folder}')
    exit(0)


# 2.每个数据动态存储 m 层
# 创建模型实例
input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 32  # 根据层数的范围确定输出层大小
predict_model = MyModel(input_size, output_size)  # 请确保input_size和output_size已定义
predict_model.to(device)
# 加载已保存的模型参数
predict_model.load_state_dict(torch.load(model_parameter,map_location=device))
models = []
mean=0

if os.path.exists(layers_folder):
    # 打开数据文件
    with open(layers_folder, 'rb') as f:
        layers = pickle.load(f)
    logging.info('步骤2--layer已加载')
    sum=0
    for i in range(len(layers)):
        sum+=layers[i]
    mean=sum/len(layers)
    logging.info(f"数组的平均值为:{mean}")
else:
    # for n in range(0, output_size ):
    #     model = imagebind_model.imagebind_huge(pretrained=True, audio_num_blocks=n)
    #     device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     model.to(device_ids[math.ceil((n-1)/2)])
    #     models.append(model)
    layers=[]
    
    for embedding_item in embeddings[ModalityType.VISION]:       
            embedding_item=embedding_item.to(device)
            layer=predict_model(embedding_item.float())
            _, layer1 = torch.max(layer, 0)
            layers.append(layer1+1)
    # 保存数据到文件
    sum=0
    for i in range(len(layers)):
        sum+=layers[i]
    mean=sum/len(layers)
    logging.info(f"数组的平均值为:{mean}")
    with open(layers_folder, 'wb') as f:
        pickle.dump(layers, f)
        
    
    logging.info('步骤2--layer已保存')




if os.path.exists(embedding_dynamic_folder):
    embedding_dynamic={}
    with torch.no_grad():
            checkpoint = torch.load(embedding_dynamic_folder)
            # 获取模型参数和张量
            embedding_dynamic[ModalityType.VISION]= checkpoint['vision_embeddings']
            
    logging.info('步骤2--dynamic存在,已加载')
else:
    embedding_dynamic={}
    with torch.no_grad():
        for i in range(len(layers)):
                #parameter_embedding_folder=f'parameters/audio/lora2/embeddings_{layers[i]}.pth'
                parameter_embedding_folder1=parameter_embedding_folder+f'embeddings_{layers[i]}.pth'
                if os.path.exists(parameter_embedding_folder1):
                    current_embeddings = torch.load(parameter_embedding_folder1, map_location=torch.device(args.device))['audio_embeddings'][i]
                    #current_embeddings=torch.load(parameter_embedding_folder)['audio_embeddings'][i]
                    if embedding_dynamic:
                            embedding_dynamic[ModalityType.VISION] = torch.cat([embedding_dynamic[ModalityType.VISION], current_embeddings.unsqueeze(0).to(embedding_dynamic[ModalityType.VISION].device)], dim=0)
                    else:
                            embedding_dynamic[ModalityType.VISION] = current_embeddings.unsqueeze(0)
                    del current_embeddings
                    
                else:
                    logging.info(f"no {parameter_embedding_folder1}")
                    logging.info('please get embeddings first')
                    exit(0)
                    
        torch.save({
                'vision_embeddings': embedding_dynamic[ModalityType.VISION]
            }, embedding_dynamic_folder)
        logging.info('步骤2--dynamic不存在,已保存')






# #3 根据query进行match到前k个数据
fine_model = imagebind_model.imagebind_huge(pretrained=True)
fine_model.modality_trunks.update(LoRA.apply_lora_modality_trunks(fine_model.modality_trunks, rank=4,
                                                                  layer_idxs={
                                                                                          ModalityType.VISION: [i for i in range(1,full_layer)]},
                                                                                modality_names=[ ModalityType.VISION]))

LoRA.load_lora_modality_trunks(fine_model.modality_trunks, checkpoint_dir=lora_dir, postfix = "_last")

load_module(fine_model.modality_postprocessors, module_name="postprocessors",
                checkpoint_dir=lora_dir)
load_module(fine_model.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir)

fine_model=fine_model.to(device)
fine_model.eval()

with open('model_architecture2.txt', 'w') as f:
    f.write(str(fine_model))
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

if os.path.exists(shortlist_folder):
    with open(shortlist_folder, 'rb') as f:
        shortlist = pickle.load(f)
        shortlist_item = pickle.load(f)
    logging.info('步骤3--shortlist存在,已加载')
    with torch.no_grad():
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(test_ds1.text_list,device=device)
            }
        fine_embeddings= fine_model(inputs)[ModalityType.TEXT]
        r1=0
        r5=0
        r10=0
        
        for batch_idx, (x, target) in enumerate(test_dl2):
                target = target.to(device)
                match_value_1 = embedding_dynamic[ModalityType.VISION][batch_idx*batch_size:batch_idx*batch_size+batch_size].to(fine_embeddings.device)@fine_embeddings.T 
                result_1 = torch.softmax(match_value_1, dim=-1)
                _, predicted = torch.max(result_1, dim=-1)
                # print(target)
                # print(predicted) 
                top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]
                
                for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                        if k == 1:
                                counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]])
                                #counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted== target.to(predicted.device))]])
                        else:
                                # print(match_value_1.shape)                            
                                # print(len(target))
                                # print(target)
                                # print(predicted)
                                counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                               
                r1=(np.sum(counts_rs['counts_r1']))/len(counts_rs['counts_r1'])
                r5=(np.sum(counts_rs['counts_r5']))/len(counts_rs['counts_r1'])
                r10=(np.sum(counts_rs['counts_r10']))/len(counts_rs['counts_r1']) 
            
                #logging.info(f"batch_idx = {batch_idx}, r1={r1},r10={r10}, test_total = {len(counts_r1)}")
               
        logging.info(f'dynamic embedding的准确率:{r1}_{r5}_{r10}')
else:    
    with torch.no_grad():
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(test_ds1.text_list,device=device)
            }
        fine_embeddings= fine_model(inputs)[ModalityType.TEXT]
        r1=0
        r5=0
        r10=0
        for batch_idx, (x, target) in enumerate(test_dl2):
                target = target.to(device)
                match_value_1 = embedding_dynamic[ModalityType.VISION][batch_idx*batch_size:batch_idx*batch_size+batch_size]@fine_embeddings.T 
                #match_value_1 = fine_embeddings @ embedding_dynamic[ModalityType.VISION].T 
                result_1 = torch.softmax(match_value_1, dim=-1)
                _, predicted = torch.max(result_1, dim=-1)
                print(target)
                print(predicted)  
                top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]
              
                for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                        if k == 1:
                                counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]])
                                #counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted== target.to(predicted.device))]])
                        else:
                                counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                                
                        for i,row in enumerate(top_indices):
                          
                            list=[]
                            list_item=[]
                            for item in row:
                                list.append(test_ds1.text_list[item])
                                list_item.append(item.item())
                            shortlist[counts_r].append(list)
                            shortlist_item[counts_r].append(list_item)
                            
                r1=(np.sum(counts_rs['counts_r1']))/len(counts_rs['counts_r1'])
                r5=(np.sum(counts_rs['counts_r5']))/len(counts_rs['counts_r1'])
                r10=(np.sum(counts_rs['counts_r10']))/len(counts_rs['counts_r1']) 
        
                file_path=shortlist_folder
                # 保存 shortlist 和 shortlist_item 到本地文件
                with open(file_path, 'wb') as f:
                    pickle.dump(shortlist, f)
                    pickle.dump(shortlist_item, f)

                logging.info(f"Data saved successfully to {file_path}")
                logging.info('步骤3--shortlist不存在,已保存')
        logging.info(f'dynamic embedding的准确率:{r1}_{r5}_{r10}')

results_dynamic=[]
lists=[]
results_dynamic.append('dynamic')
results_dynamic.append(N)
results_dynamic.append(K)
results_dynamic.append(S)
results_dynamic.append(mean)
for counts in counts_rs:
    correct=np.sum(counts_rs[counts] == 1)/len(counts_rs['counts_r1'])
    results_dynamic.append(str(correct))
    lists.append(str(counts))
    
# exit(0)

# 4 再次进行fine-grained embedding
# num_samples = 5000
batch_size=1
test_dl3 = DataLoader(dataset=subset_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)

# batch_size=1
# test_dl2 = DataLoader(dataset=test_ds1, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, persistent_workers=True)
# test_dl2 = itertools.islice(test_dl2, num_samples)
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
        checkpoint = torch.load(fine_model_embeddings)
        # 获取模型参数和张量
        embeddings_all[ModalityType.VISION]= checkpoint['audio_embeddings']
        logging.info(f"step 4: fine-grained embedding")
        r1=0
        r5=0
        r10=0
        for batch_idx, (x, target) in enumerate(test_dl3):
            # print(batch_idx)
            embeddings_TEXT={}
            text_list=[]
            x = x.to(device)
            
            target=[t.to(device) for t in target]
            
           
            for item in shortlist_item[f'counts_r{K}'][batch_idx*batch_size]:
                       text_list.append(test_ds1.text_list[item])
           
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(text_list, device)
                #ModalityType.VISION: data.load_and_transform_audio_data(shortlist[f'counts_r{K}'][batch_idx*batch_size],device=device)
            }
            #fine_embeddings= fine_model(inputs)[ModalityType.TEXT].to(embedding_dynamic[ModalityType.VISION].device)
            
            embeddings_all = fine_model(inputs)
            match_value_1 = embeddings_all[ModalityType.VISION]@embeddings_all[ModalityType.TEXT].T 
            
            #match_value_1 = embedding_dynamic[ModalityType.VISION][batch_idx]@fine_embeddings.T
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, dim=-1)
            _, topk_indices = torch.topk(result_1, k=K, dim=-1)
            
            top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] if k <= K else None for k in topk1]
            
            predicted=torch.Tensor([shortlist_item[f'counts_r{K}'][batch_idx*batch_size][predicted]])
         
            for k, top_indices, counts_rk in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                       # counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]])
                                
                        counts_rs[counts_rk] = np.concatenate([counts_rs[counts_rk], [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                elif k<=K:            
                        results_k=[]
                        for i_k in range(k):
                            results_k.append(shortlist_item[f'counts_r{K}'][batch_idx][top_indices[0][i_k].item()])
                        counts_rs[counts_rk] = np.concatenate([counts_rs[counts_rk], [int(any(results_k[i] == target[0].to(predicted.device))) for i in range(len(results_k))]])
                        
                else :
                    break
            r1=(np.sum(counts_rs['counts_r1']))/len(counts_rs['counts_r1'])
            r5=(np.sum(counts_rs['counts_r5']))/len(counts_rs['counts_r1'])
            r10=(np.sum(counts_rs['counts_r10']))/len(counts_rs['counts_r1']) 
        logging.info(f"fine-grained embedding : {r1}_{r5}_{r10}")
        
results=[]
lists=[]
results.append('total') 
results.append(N)
results.append(K)
results.append(S)
results.append(mean)
for counts in counts_rs:
    correct=np.sum(counts_rs[counts] == 1)/len(counts_rs['counts_r1'])
    results.append(str(correct))
    lists.append(str(counts))
lists.append(['N'])
lists.append(['K'])
lists.append(['S'])
import csv

# 数据
data1 = [
    results,
    results_dynamic
]

# # 指定CSV文件路径
csv_file_path = f'end_to_end_lora_N_K_S_{split}.csv'

with open(csv_file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入新数据
    for row in data1:
        writer.writerow(row)
        
        

