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
csv_file_path = "/data/air/pc/Mobile-Search-Engine/datasets/clotho/clotho_captions_evaluation.csv"
data_dir="/data/air/pc/Mobile-Search-Engine/datasets/clotho/evaluation"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=["/data/air/pc/Mobile-Search-Engine/datasets/clotho/evaluation/"+file for file in audio_list]

# import argparse
# # 创建解析器
# parser = argparse.ArgumentParser(description="Your script description")
# # 添加命令行参数
# #parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")
# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
# #parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda:2 or cpu)")
# #parser.add_argument("--text_num_blocks", default=24,type=int, help="Number of audio blocks")
# # 解析命令行参数
# args = parser.parse_args()
# # 获取 audio_num_blocks 的值
# audio_num_blocks = args.audio_num_blocks
#text_num_blocks=args.text_num_blocks
#device = args.device

audio_num_blocks=12

device_ids = [4,5,6,7] 
device = "cuda:5" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)


#
model = DataParallel(model)
model=model.cuda()
model.eval()

import pandas as pd
def run_inference():
    # csv_file_path = "/home/pc/Mobile-Search-Engine/datasets/clotho_captions_evaluation2.csv"
    #data_dir="/home/pc/Mobile-Search-Engine/datasets/evaluation"
    # pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
    Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
    test_dl = DataLoader(dataset=Clotho_dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
    counts_r1=np.array([])
    counts_r10=np.array([])
    count_ones_r10=0
    audio_dataset=ClothoDataset(csv_file=csv_file_path,device=device,datadir=data_dir)

    audio_dl=DataLoader(dataset=audio_dataset,batch_size=64, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
    
    batch_size=320
    batches=[audio_path[i:i+batch_size] for i in range(0,len(audio_path),batch_size)]
    audio_embeddings=torch.Tensor().to(device)
    # with torch.no_grad():
    #         input={
    #             ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=audio_embeddings.device)
    #             #ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=audio_embeddings.device)
    #         }
    #         audio_embedding_t=model(input)
            
    # with torch.no_grad():
    #     for batch in batches:
    #         input={
    #             ModalityType.AUDIO: data.load_and_transform_audio_data(batch,device=audio_embeddings.device)
    #             #ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=audio_embeddings.device)
    #         }
    #         audio_embedding=model(input)
    #         audio_embeddings=torch.cat((audio_embeddings,audio_embedding[ModalityType.AUDIO].to(audio_embeddings.device)),dim=0)
    # #audio_embeddings=audio_embedding
        
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            target = target.to(device)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(x, device),
                ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=audio_embeddings.device)
            }

            embeddings = model(inputs)
            #match_value_1 = embeddings[ModalityType.TEXT].to(audio_embeddings.device)@audio_embeddings.T 
            #match_value_1 = embeddings[ModalityType.TEXT] @ embeddings[ModalityType.AUDIO].T 
            match_value_1 = embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T 
            result_1 = torch.softmax(match_value_1, dim=0)
            _, predicted = torch.max(result_1, dim=0)
            _, topk_indices = torch.topk(result_1, k=10, dim=0)
            counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
            #counts_r1 = np.concatenate([counts_r1, [any(predicted[i] == target[i]) for i in range(len(predicted))]])
            topk_indices=topk_indices.T
            counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            # counts_10= calculate_intersection(target, topk_indices)
            # counts_10_ratio=[x / 5 for x in counts_10]
            #counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i]) for i in range(len(predicted)))]])
            #counts_r10 = np.concatenate([counts_r10, [any(np.any(topk_indices[i] == target[i]) for i in range(len(predicted)))]])
            # for i in range(len(predicted)):
            #     counts_r10.append(any(np.any(topk_indices[i] == target[i])))
            #counts_r10 = np.concatenate([counts_r10,[len(np.intersect1d(topk_indices[i].cpu().numpy(), target[i].cpu().numpy()))>0 for i in range(len(predicted))]])
            #counts_r10 = np.array([np.any(topk_indices[i] == target[i]) for i in range(len(predicted))])
            r1=(np.sum(counts_r1==1))/len(counts_r1)
            r10=(np.sum(counts_r10==1))/len(counts_r10) 
          
            logging.info(f"batch_idx = {batch_idx}, r1={r1},r10={r10}, test_total = {len(counts_r1)}")


    #logging.info(f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {acc}, Recall = {recall}")
    #print(len(counts_r10))
    count_ones_r1 = np.sum(counts_r1 == 1)
    count_ones_r10 = np.sum(counts_r10 == 1)
    r1=count_ones_r1/len(counts_r1)
    r10=count_ones_r10/len(counts_r1)
    np.savetxt(f'./results/clotho_t2a/R1/t{t_block}_a{a_block}_acc{r1}.txt',counts_r1,fmt='%d')
    np.savetxt(f'./results/clotho_t2a/R10/t{t_block}_a{a_block}_acc{r10}.txt',counts_r10,fmt='%d')
    
    return r1,r10

def main():
    Accuracy = run_inference()
    print("Model Performance:", Accuracy)

def print_text_label():
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

    datadir = "./.datasets/imagenet"
    test_ds = ImageNet(datadir, split="val", transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    
    labels = sorted(list(set(batch[1] for batch in test_dl)))
    print(labels)

if __name__ == "__main__":
    main()
    # print_text_label()
