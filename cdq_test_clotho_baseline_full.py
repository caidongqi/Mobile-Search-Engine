import logging
import torch
import data
import argparse
import numpy as np
import os
import pandas as pd
import json
import math
import subprocess
import re


from models import imagebind_model
from models.imagebind_model import ModalityType, load_module

from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

logging.basicConfig(level=logging.INFO, force=True)

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns the item at the given index `idx`.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            Tensor: The audio sample at the given index.
        """
        audio_sample = self.data[idx]
        return audio_sample
    
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # 输入层到隐藏层
        self.fc2 = nn.Linear(256, 64)  # 隐藏层到输出层
        self.fc3 = nn.Linear(64, output_size)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def grouping(predicted):
    # 根据predicted结果创建索引分组
    groups = defaultdict(list)
    for idx, pred in enumerate(predicted):
        groups[pred.item()].append(idx)
    
    return groups

def cal_avg_layer(group):
    # 计算每个 item 长度和其 key 值相乘的均值
    total_length_times_key = 0
    total_items = 0

    for key, item in group.items():
        total_length_times_key += len(item) * int(key)
        total_items += len(item)

    average_length_times_key = total_length_times_key / total_items
    return average_length_times_key

device = "cuda:7" if torch.cuda.is_available() else "cpu"

input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 12  # 根据层数的范围确定输出层大小
new_model = MyModel(input_size, output_size)

# 加载之前保存的模型参数
new_model.load_state_dict(torch.load('/data/air/pc/Mobile-Search-Engine/model_trunks1&2_parameters.pth', map_location=device))
new_model.to(device)

# eval data
csv_file_path = "/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_csv_files/clotho_captions_evaluation.csv"
data_dir="/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_audio_files/evaluation"
pf = pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
audio_list = pf[['file_name']].values.flatten().tolist()
audio_path = [os.path.join(data_dir,file) for file in audio_list]

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True, elastic=True)

model.eval()
model.to(device)

if (os.path.exists('audioData.npy')):
    audioData = torch.tensor(np.load('audioData.npy')).to(device)
    print('load audioData done!')
else:
    audioData = data.load_and_transform_audio_data(audio_path,device)
    np.save('audioData.npy',audioData.cpu().numpy())
    print('transform audioData done!')

def run_inference():
    audioDataset = AudioDataset(audioData)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_latency = []
    gpu_utlization_list = []
    warm_start = 3

    def get_gpu_utilization_nvidia_smi():
        # 使用nvidia-smi命令获取GPU状态信息
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], encoding='utf-8')
        
        # 解析输出以获得GPU使用率，nvidia-smi通常返回一个百分比值
        gpu_utilization_match = re.search(r'\d+', nvidia_smi_output)
        if gpu_utilization_match:
            gpu_utilization = int(gpu_utilization_match.group())
            return gpu_utilization
        else:
            # 如果解析失败，返回None或其他适当的错误处理方式
            return None

    def embedding(dataset, ee=100, batch_size=1, re_enter=0):
        embeddings_all = torch.empty(0,1024).to(device)
        with torch.no_grad():
            start_event.record()
            test_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0, pin_memory=False, persistent_workers=False)
            end_event.record()
            torch.cuda.synchronize()
            logging.info(f"{{BS={batch_size}}} test_dl creation latency = {start_event.elapsed_time(end_event)}ms")
            for batch_idx, x in enumerate(test_dl):
                start_event.record()

                x = x.to(device)
                inputs = {
                    ModalityType.AUDIO: x
                }

                
                embeddings = model(inputs, ee, re_enter)[ModalityType.AUDIO]
                embeddings_all = torch.cat((embeddings_all, embeddings), dim = 0)
                
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)

                total_latency.append(elapsed_time)
                
                gpu_utilization = get_gpu_utilization_nvidia_smi()

                gpu_utlization_list.append(gpu_utilization)                

                if batch_idx % 10 == 0:
                    logging.info(f"{{BS={batch_size}, ee={ee}}} process = {batch_idx}/{len(test_dl)}, batch_latency={elapsed_time}ms, sample_latency = {np.array(total_latency[warm_start:]).sum() / ((batch_idx + 1 - warm_start) * batch_size)}ms, total_latency = {np.array(total_latency[:]).sum() / 1000 :3f}s, GPU Utilization = {gpu_utilization}%")
        return embeddings_all
    
    embeddings_all = embedding(audioDataset, ee, batch_size)

    return np.array(total_latency).sum() / 1000

parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--bs', type=int, default=32, help='Batch size for testing')
parser.add_argument('--ee', type=int, default=100, help='Early exit layers')
args = parser.parse_args()
batch_size = args.bs
ee = args.ee

latency = run_inference()
print("Total latency:", latency)
