import logging
import torch
import data
import torchvision
import torchmetrics
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
# eval data
csv_file_path = "/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_csv_files/clotho_captions_evaluation.csv"
data_dir="/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_audio_files/evaluation"
# train data
# csv_file_path = "/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_csv_files/clotho_captions_development.csv"
# data_dir="/data/yx/MobileSearchEngine/Mobile-Search-Engine-main/.datasets/data/clotho_audio_files/development"


f_s=os.listdir(data_dir)
print('test data:',len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=[os.path.join(data_dir,file) for file in audio_list]

device = "cuda:4" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
lora=True
load_head_post_proc_finetuned = True
model = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=6,vision_num_blocks=0)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.AUDIO: [ 1, 2,3,4,5]},
                                        modality_names=[ModalityType.AUDIO]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir="./.checkpoints/lora/clotho_6")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir="./.checkpoints/lora/clotho_6")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir="./.checkpoints/lora/clotho_6")


#
# model = DataParallel(model)

model.eval()
model.to(device)

import pandas as pd
import tqdm
def run_inference():
    # csv_file_path = "/home/pc/Mobile-Search-Engine/datasets/clotho_captions_evaluation2.csv"
    #data_dir="/home/pc/Mobile-Search-Engine/datasets/evaluation"
    # pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
    print('loading dataset')
    Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
    test_dl = DataLoader(dataset=Clotho_dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
    counts_r1=0
    counts_r10=0
    total=0

    print('start inference')
    with torch.no_grad():
        print('start audio embeddings')

        if (os.path.exists('audioData.npy')):
            audioData = torch.tensor(np.load('audioData.npy')).to(device)
            print('load audioData done!')
        else:
            audioData = data.load_and_transform_audio_data(audio_path,device)
            np.save('audioData.npy',audioData.cpu().numpy())
            print('transform audioData done!')


        total_batches = (audioData.size(0) + 31) // 32
    
        all_outputs = []  # 存储所有批次的输出
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * 32
            end_idx = start_idx + 32
            data_batch = audioData[start_idx:end_idx]
            
            # 模型预测
            with torch.no_grad():  # 确保不会计算梯度
                batch_output = model({ModalityType.AUDIO: data_batch})[ModalityType.AUDIO]
            
            all_outputs.append(batch_output)
        
        # 使用torch.cat聚合输出，这自动处理了最后一个批次可能小于batch_size的情况
        audio_embeddings = torch.cat(all_outputs, dim=0)
        print('audio_embeddings done!')

        for (x, target) in tqdm.tqdm(test_dl):
            target = target.to(device)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(x, device),
            }

            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.TEXT] @ audio_embeddings.T 
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, dim=-1)
            _, topk_indices = torch.topk(result_1, k=10, dim=-1)
            counts_r1 += torch.sum(predicted == target).item()
            #counts_r1 = np.concatenate([counts_r1, [any(predicted[i] == target[i]) for i in range(len(predicted))]])
            topk_indices=topk_indices.T
            counts_r10 += torch.sum(topk_indices == target).item()

            total += len(target)
            
            r1=counts_r1/total
            r10=counts_r10/total
          
            logging.info(f"r1={r1},r10={r10}, test_total = {total}")


    #logging.info(f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {acc}, Recall = {recall}")
    #print(len(counts_r10))
    
    return r1,r10


if __name__ == "__main__":
    Accuracy = run_inference()
    print("Model Performance:", Accuracy)
