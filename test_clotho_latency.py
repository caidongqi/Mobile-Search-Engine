import logging
import torch
import data
import argparse
import numpy as np
import os
import pandas as pd
import subprocess
import re

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module

from torch.utils.data import Dataset, Subset, DataLoader

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

device = "cuda:4" if torch.cuda.is_available() else "cpu"

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
    warm_start = 3

    def get_gpu_utilization_nvidia_smi():
        # 使用nvidia-smi命令获取GPU状态信息
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], encoding='utf-8')
        
        # 解析输出以获得GPU使用率，nvidia-smi通常返回一个百分比值
        gpu_utilization_match = re.search(r'\d+', nvidia_smi_output)
        if gpu_utilization_match:
            gpu_utilization = int(gpu_utilization_match.group())
            return gpu_utilization
        else:
            # 如果解析失败，返回None或其他适当的错误处理方式
            return None

    def embedding(dataset, ee=100, batch_size=1, re_enter=0):
        with torch.no_grad():
            start_event.record()
            test_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=4, pin_memory=False, persistent_workers=False)
            end_event.record()
            torch.cuda.synchronize()
            logging.info(f"{{BS={batch_size}}} test_dl creation latency = {start_event.elapsed_time(end_event)}ms")
            for batch_idx, (x) in enumerate(test_dl):
                start_event.record()

                x = x.to(device)
                inputs = {
                    ModalityType.AUDIO: x
                }

                
                _ = model(inputs, ee, re_enter)[ModalityType.AUDIO]
                
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)

                total_latency.append(elapsed_time)
            
                if batch_idx % 10 == 0:
                    logging.info(f"{{BS={batch_size}, ee={ee}}} process = {batch_idx}/{len(test_dl)}, batch_latency={elapsed_time}ms, sample_latency = {np.array(total_latency[warm_start:]).sum() / ((batch_idx + 1 - warm_start) * batch_size)}ms, total_latency = {np.array(total_latency[:]).sum() / 1000 :3f}s")

    
    # Step 1: we fix ee = 1 and predict the ee of all samples
    ee = 1
    embedding(audioDataset, ee, batch_size)
    
    # Step 2: we sort the most confident ee and batch them accoradingly. ee = 1 and predict the ee of all samples        
    sorted_batch_idx_1 = np.argsort(total_latency[:])
    num_samples = int(1000 * 1.0)
    # Group samples with same ee and batch them together
    selected_batch_idx_1 = {
        2: range(0,100),
        3: range(100,300),
        4: range(300,600),
        5: range(600,1000)
    }

    re_enter = 1
    for ee in selected_batch_idx_1:
        indices = selected_batch_idx_1[ee]
        subset_ds = Subset(audioDataset, indices)
        embedding(subset_ds, ee, batch_size, re_enter)
        
    ## Step 3: We compute the second layer of the rest samples and predict their ee. Repeat it until all samples are processed.
        
    return np.array(total_latency).sum() / 1000

parser = argparse.ArgumentParser(description='Test script')
parser.add_argument('--bs', type=int, default=32, help='Batch size for testing')
parser.add_argument('--ee', type=int, default=100, help='Early exit layers')
args = parser.parse_args()
batch_size = args.bs
ee = args.ee

latency = run_inference()
print("Total latency:", latency)
