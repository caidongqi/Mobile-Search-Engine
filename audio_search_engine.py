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
csv_file_path = "/home/u2021010261/data/cdq/clotho/clotho_captions_evaluation.csv"
data_dir="/home/u2021010261/data/cdq/clotho/evaluation"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=["/home/u2021010261/data/cdq/clotho/evaluation/"+file for file in audio_list]
import random
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--audio_num", default=1,type=int, help="Number of audio blocks")
parser.add_argument("--text_num_blocks", default=24,type=int, help="Number of text blocks")

# 解析命令行参数
args = parser.parse_args()

# 获取 audio_num_blocks 的值
audio_num_blocks=args.audio_num
text_num_blocks=args.text_num_blocks
device = args.device

audio_num_blocks_1=audio_num_blocks
audio_num_blocks_2=12
device_ids = [1,2,3,4,5] 
device = "cuda:6" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_1 = imagebind_model.imagebind_huge(pretrained=True,text_num_blocks=text_num_blocks,audio_num_blocks=audio_num_blocks)
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)
v_block=len(model_1.modality_trunks["vision"].blocks)
t_block=len(model_1.modality_trunks["text"].blocks)
a_block=len(model_1.modality_trunks["audio"].blocks)
i_block=len(model_1.modality_trunks["imu"].blocks)



lora=False
linear_probing=False
load_head_post_proc_finetuned=False
lora_dir =f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/clotho/step1/epoch50/{audio_num_blocks}'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 256 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1
#device = "cuda:0" if torch.cuda.is_available() else "cpu"

if lora:
    model_1.modality_trunks.update(LoRA.apply_lora_modality_trunks(model_1.modality_trunks, rank=4,
                                                                              layer_idxs=None,
                                                                              modality_names=[ModalityType.TEXT, ModalityType.AUDIO]))
   
    LoRA.load_lora_modality_trunks(model_1.modality_trunks, checkpoint_dir=lora_dir, postfix = "_last")
           

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model_1.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir)
        load_module(model_1.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir)
elif linear_probing:
    # Load heads
    load_module(model_1.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir)
   
model_1=model_1.cuda()
model_1 = model_1.to(device) 
model_1.eval()

# model_2.eval()
# model_2.to(device)
import pandas as pd
def run_inference():
    Clotho_dataset = ClothoTextDataset(csv_file=csv_file_path,device=device)
    batch_size=128
    test_dl = DataLoader(dataset=Clotho_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
    counts_r1=np.array([])
    counts_r10=np.array([])
    count_ones_r10=0
    batches=[audio_path[i:i+batch_size] for i in range(0,len(audio_path),batch_size)]
    batch=audio_path[0:29]
    audio_embeddings=torch.Tensor().to(device)
    topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600,700,800,900,1000]
    counts_rs = {}
    shortlist={}
    shortlist_item={}
    for k in topk1:
                counts_rs[f'counts_r{k}'] = np.array([])
                shortlist[f'counts_r{k}']=[]
                shortlist_item[f'counts_r{k}']=[]
    embeddings={}
    with torch.no_grad():
        
            checkpoint = torch.load(f'parameters/audio/lora/embeddings_{a_block}_trunks.pth')
            # 获取模型参数和张量
            embeddings[ModalityType.AUDIO]= checkpoint['audio_embeddings']
            print(1)


    with torch.no_grad():
        
        for batch_idx, (x, target) in enumerate(test_dl):
            target = target.to(device)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(x, device)
            }

            embeddings_text = model_1(inputs)
            #match_value_1 = embeddings[ModalityType.TEXT].to(audio_embeddings.device)@audio_embeddings.T 
            #match_value_1 = embeddings[ModalityType.TEXT] @ embeddings[ModalityType.AUDIO].T 
            embeddings[ModalityType.AUDIO]=embeddings[ModalityType.AUDIO].to(embeddings_text[ModalityType.TEXT].device)
            match_value_1 = embeddings_text[ModalityType.TEXT] @ embeddings[ModalityType.AUDIO].T 
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, dim=-1)
            _, topk_indices = torch.topk(result_1, k=10, dim=-1)
            counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
            counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            
            top_indices_list = [torch.topk(result_1, k=k, dim=-1)[1] for k in topk1]
            
            for k, top_indices, counts_r in zip(topk1, top_indices_list, counts_rs):
                if k == 1:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                    
                else:
                    counts_rs[counts_r] = np.concatenate([counts_rs[counts_r], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
           
            
            r1=(np.sum(counts_rs['counts_r1']==1))/len(counts_rs['counts_r1'])
            r5=(np.sum(counts_rs['counts_r5']==1))/len(counts_rs['counts_r1']) 
            r10=(np.sum(counts_rs['counts_r10']==1))/len(counts_rs['counts_r1']) 
            
            logging.info(f"batch_idx = {batch_idx}, r1={r1},r10={r5}, test_total = {len(counts_rs['counts_r1'])}")
        np.savetxt(f'./results/clotho/lora/text_nohead/R10/t{t_block}_a{a_block}.txt',counts_r10,fmt='%d')
        np.savetxt(f'./results/clotho/lora/text_nohead/R1/t{t_block}_a{a_block}.txt',counts_r1,fmt='%d')

    
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
