import logging
import torch
import data
import torchvision
import torchmetrics
import torch.nn as nn
from models import imagebind_model
from models import imagebind_model_only_trunks
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

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
#parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--audio_num", default=3,type=int, help="Number of audio blocks")
# 解析命令行参数
args = parser.parse_args()

# 获取 audio_num_blocks 的值
audio_num_blocks=args.audio_num

audio_num_blocks_1=audio_num_blocks
audio_num_blocks_2=12
lora=True
linear_probing=False
load_head_post_proc_finetuned=True
lora_dir =f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/clotho/step1/{audio_num_blocks}'
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
model_1 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_1)

if lora:
    model_1.modality_trunks.update(LoRA.apply_lora_modality_trunks(model_1.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.AUDIO: [i for i in range(1,audio_num_blocks_1)]},
                                        modality_names=[ModalityType.AUDIO]))
 
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
    
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)

model_1=model_1.cuda()
model_1 = model_1.to(device) 


model_1.eval()

with open('model_architecture3.txt', 'w') as f:
    f.write(str(model_1))


# model_2.eval()
# model_2.to(device)
embeddings={}
import pandas as pd
with torch.no_grad():
    
    input={  
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path,device=device)
    }
    embeddings[ModalityType.AUDIO] = model_1(input)[ModalityType.AUDIO]
    # 假设你想要保存模型参数和张量
    torch.save({
        'audio_embeddings': embeddings[ModalityType.AUDIO]
    }, f'parameters/audio/lora2/embeddings_{audio_num_blocks}.pth')