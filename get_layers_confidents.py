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
from api.clotho import ClothoDataset
from api.coco_text2image import CoCo_t2i_Dataset
import os
import csv
import argparse
from api.flickr import flickr8k
import pickle

# # 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--vision_num_blocks", type=int,default=32, help="Number of vision blocks")
parser.add_argument("--version", type=str, default='flicker_confidence', help="version of test lora")
parser.add_argument("--lora_dir", type=str, default='/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/flickr8k/without_head/trunk/e100/{vision_num_blocks}', help="lora dir")
parser.add_argument("--result_path", type=str, default='./results/flickr8k_confidence', help="infer results dir")
parser.add_argument("--csv_file_path", type=str, default='flickr8k_confidence.csv', help="infer output csv path")


args = parser.parse_args()
vision_num_blocks=args.vision_num_blocks
version=args.version
lora_dir=args.lora_dir
lora_dir=lora_dir.format(vision_num_blocks=vision_num_blocks)
result_path = args.result_path

csv_file_path = args.csv_file_path

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
                    handlers=[logging.FileHandler(f'logs/infer/flickr8k{vision_num_blocks}_{version}_{formatted_time}.log')], 
                    force=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned=True

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=vision_num_blocks)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)

# Load fine-tuned text heads
load_module(model.modality_heads, module_name="heads",
            checkpoint_dir=lora_dir, device =device)

model.to(device)
model.eval()

datadir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.datasets/flickr8k/images"
anne_dir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.datasets/flickr8k/captions.txt"
test_ds = flickr8k(root_dir=datadir, anne_dir=anne_dir, split='test')
test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, drop_last=False,
num_workers=4, pin_memory=True, persistent_workers=True)
batch_size=1
text_prompt = 'a photo of {}.'
# 从文件加载字典
with open('/home/u2021010261/data/yx/Mobile-Search-Engine-main/flickr8k_img_dict.pkl', 'rb') as file:
    img_dict = pickle.load(file)

text_embeddings_dir='parameters/image/flickr8k/val/text_embeddings_N=2_S=10_ground_truth_flicker.pt'
coarse_embedding_path=f'parameters/image/flickr8k/val/embeddings_{vision_num_blocks}.pth'

import pandas as pd
def run_inference():    
    all_text_embeddings = torch.load(text_embeddings_dir, map_location=torch.device(device))
    logging.info('text_embeddings存在,已加载')

    image_embeddings={} # load from coarse_embedding_path
    if os.path.exists(coarse_embedding_path):
        with torch.no_grad():
            checkpoint = torch.load(coarse_embedding_path)
            # 获取模型参数和张量
            image_embeddings[ModalityType.VISION]= checkpoint['vision_embeddings']
            logging.info('vision embedding已加载')

    all_confidence=[]
    with torch.no_grad():
        for batch_idx, (_, x, image_name) in enumerate(test_dl):
            text_embedding= all_text_embeddings[batch_idx*batch_size]
            image_embedding= image_embeddings[ModalityType.VISION][batch_idx*batch_size]
            confidence=text_embedding@image_embedding.T
            all_confidence.append(confidence.cpu())
            logging.info(f"batch_idx = {batch_idx}, confidence={confidence}")
        
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        file_path=os.path.join(result_path,f'V={v_block}.txt')
        np.savetxt(file_path,all_confidence,fmt='%d')
    
    mean_confidence=np.mean(all_confidence)

    results=['vison_layer','mean_confidence']
    co_results=[v_block, mean_confidence]
    data1 = [results, co_results]

    # # 指定CSV文件路径

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入新数据
        for row in data1:
            writer.writerow(row)
    return mean_confidence

def main():
    Accuracy = run_inference()
    print("Model Performance:", Accuracy)


if __name__ == "__main__":
    main()
   
