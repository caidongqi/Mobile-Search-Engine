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
logging.basicConfig(level=logging.INFO, force=True)
import os
import csv
import argparse
# # 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--vision_num_blocks", type=int,default=32, help="Number of audio blocks")

args = parser.parse_args()


vision_num_blocks=args.vision_num_blocks

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=vision_num_blocks)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)


#
model = DataParallel(model)
model=model.cuda()
model.eval()

# coco_datadir="/home/u2021010261/share/pc/COCO/val2017"
# coco_annotation_file='/home/u2021010261/share/pc/COCO/instances_val2017.json'
# test_ds=CoCoDataset(datadir=coco_datadir, annFile=coco_annotation_file,transform=data_transform)
# test_dl=DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

coco_annotation_file = "/home/u2021010261/share/pc/COCO/captions_val2017.json"
data_dir="/home/u2021010261/share/pc/COCO/val2017"
CoCo_dataset = CoCo_t2i_Dataset(json_file=coco_annotation_file,datadir=data_dir,device=device)
test_dl = DataLoader(dataset=CoCo_dataset, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

coco_embedding_path=f'parameters/image/coco/embeddings_{v_block}.pth'
import pandas as pd
def run_inference():    
    topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]
    counts_rs = {}
    for k in topk1:
                counts_rs[f'counts_r{k}'] = np.array([])
    
    with torch.no_grad():
        checkpoint = torch.load(coco_embedding_path)
        vision_embeddings= checkpoint['vision_embeddings'] # TODO: audio_embeddings -> xx_embedding
        all_predictions = []
        for batch_idx, (x, target) in enumerate(test_dl):
            target = target.to(device)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(x, device)
                
            }

            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.TEXT].to(vision_embeddings.device)@vision_embeddings.T 
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            all_predictions.append(predicted)
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
        all_predictions = torch.cat(all_predictions)
        torch.save(all_predictions, f'parameters/imagebind_targets/imagebind_{vision_num_blocks}.pt')
        for k in topk1:
            path=f'./results/coco/R{k}'
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            file_path=os.path.join(path,f'v{v_block}_t{t_block}.txt')
            np.savetxt(file_path,counts_rs[f'counts_r{k}'],fmt='%d')
        
    results=['vison_layer','R1','R5','R10']
    co_results=[v_block, r1, r5, r10]
    data1 = [results, co_results]

    # # 指定CSV文件路径
    csv_file_path = f'test_coco_zeroshot.csv'

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
