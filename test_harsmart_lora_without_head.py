# Original File: train_clotho.py
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
from api.harsmart import HARsmart_relative
from api.coco_text2image import CoCo_t2i_Dataset
import os
import csv
import argparse
# # 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--vision_num_blocks", type=int,default=6, help="Number of vision blocks")
parser.add_argument("--version", type=str, default='harsmart_withou_head', help="version of test lora")
parser.add_argument("--lora_dir", type=str, default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/harsmart/without_head/trunk/e50/', help="lora dir")
parser.add_argument("--embedding_path", type=str, default='parameters/image/harsmart/val/without_head/embeddings_{i_block}.pth', help="embeddings dir")
parser.add_argument("--result_path", type=str, default='./results/harsmart_lora_val_without_head', help="infer results dir")
parser.add_argument("--csv_file_path", type=str, default='test_harsmart_tlora_lora_without_head.csv', help="infer output csv path")


args = parser.parse_args()
vision_num_blocks=args.vision_num_blocks
version=args.version
lora_dir=args.lora_dir
lora_dir=f'{lora_dir}/{vision_num_blocks}'
embedding_path= args.embedding_path
csv_file_path = args.csv_file_path
result_path=args.result_path

import time
# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间
local_time = time.localtime(timestamp)
# 格式化本地时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'logs/infer/{vision_num_blocks}_{version}_{formatted_time}.log')], 
                    force=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned=True

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True,imu_num_blocks=vision_num_blocks)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)

embedding_path=embedding_path.format(i_block=i_block)
# Load fine-tuned text heads
load_module(model.modality_heads, module_name="heads",
            checkpoint_dir=lora_dir, device =device)

model.to(device)
model.eval()

global_batch_size = 256
test_ds = HARsmart_relative(train=False)
test_dl = DataLoader(dataset=test_ds, batch_size=global_batch_size, shuffle=False, drop_last=False,
num_workers=4, pin_memory=True, persistent_workers=True)


import pandas as pd
def run_inference():    
    topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]
    counts_rs = {}
    for k in topk1:
        counts_rs[f'counts_r{k}'] = np.array([])
    with torch.no_grad():
        checkpoint = torch.load(embedding_path, map_location=device)
        vision_embeddings= checkpoint['vision_embeddings'] # TODO: audio_embeddings -> xx_embedding
        true_embeddings=torch.load('harsmart_val_imu_embedding_full_zeroshot.pt',map_location=device)
        for batch_idx, (_, target) in enumerate(test_dl):
            target = target.to(device)
            embeddings =torch.cat([true_embeddings[i].unsqueeze(dim=0) for i in target.tolist()],dim=0)
            match_value_1 = embeddings.to(vision_embeddings.device)@vision_embeddings.T 
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]
            
            for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                else:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            
            #logging.info(f"batch_idx = {batch_idx}, test_correct = {np.sum(counts_rs['counts_r1'] == 1)/len(counts_rs['counts_r1'])}, test_total = {np.sum(counts_rs['counts_r5'] == 1)/len(counts_rs['counts_r1'])}, Accuracy = {np.sum(counts_rs['counts_r10'] == 1)/len(counts_rs['counts_r1'])}")
            data_length=len(counts_rs['counts_r1'])
            r1=(np.sum(counts_rs['counts_r1']))/data_length
            r5=(np.sum(counts_rs['counts_r5']))/data_length
            r10=(np.sum(counts_rs['counts_r10']))/data_length

            logging.info(f"batch_idx = {batch_idx}, r1={r1},r5={r5},r10={r10}, test_total = {data_length}")
        for k in topk1:
            path=f'{result_path}/R{k}'
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            file_path=os.path.join(path,f'i{i_block}.txt')
            np.savetxt(file_path,counts_rs[f'counts_r{k}'],fmt='%d')
        
    results=['vison_layer','R1','R5','R10']
    co_results=[v_block, r1, r5, r10]
    data1 = [results, co_results]

    # # 指定CSV文件路径

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入新数据
        for row in data1:
            writer.writerow(row)
    return r1,r5,r10

def main():
    Accuracy = run_inference()
    print("Model Performance:", Accuracy)


if __name__ == "__main__":
    main()
    # print_text_label()
