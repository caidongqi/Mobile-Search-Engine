import logging
import torch
import data
import torchvision
import torchmetrics
import time
import csv
import itertools
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora_lumen as LoRA

from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from api.imagenet import ImageNetDataset
from api.coco import CoCoDataset
from metrics.accuracy import Accuracy
from torch.nn.parallel import DataParallel
import openpyxl
logging.basicConfig(level=logging.INFO, force=True)
import numpy as np
print("test_imagebet")
lora = True
linear_probing = False
load_head_post_proc_finetuned = True
imagenet_datadir = "/home/u2021010261/data/yx/imagenet"
#coco_datadir="/data/air/pc/ImageBind/dataset/val2017/val2017"
# test_coco="/data/air/pc/ImageBind/dataset/tempoimage"
# datadir1='/data/air/pc/Mobile-Search-Engine/datasets/imagenet-10'
# datadir2="/data/air/pc/i-Code/i-Code-V3/dataset/tempo-10"
# coco_annotation_file='/data/air/pc/ImageBind/dataset/annotations_trainval2017/annotations/instances_val2017.json'
# lora_dir = '/data/air/pc/Mobile-Search-Engine/.checkpoints/550_epochs_lora'

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--lora_layers", default=0,type=int, help="Number of audio blocks")
parser.add_argument("--vision_num_blocks", default=26,type=int, help="Number of vision blocks")
parser.add_argument("--split", default='train',type=str, help="train or val")

# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
lora_layers=args.lora_layers
vision_num_blocks=args.vision_num_blocks
split=args.split
audio_num_blocks_2=12
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_1 = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=lora_layers+1)
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)
v_block=len(model_1.modality_trunks["vision"].blocks)
t_block=len(model_1.modality_trunks["text"].blocks)
a_block=len(model_1.modality_trunks["audio"].blocks)
i_block=len(model_1.modality_trunks["imu"].blocks)
model_1=model_1.cuda()
model_1 = model_1.to(device) 
model_1.eval()

def run_inference():
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
    
    #test_ds1=CoCoDataset(datadir=coco_datadir, annFile=coco_annotation_file,transform=data_transform)
    test_ds = ImageNetDataset(datadir=imagenet_datadir, split=split, transform=data_transform)
    #test_dl1=DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
    batch_size=64
   
    test_dl2=test_dl
    import os
    test_acc = Accuracy(task="multiclass", num_classes=1000, average="micro",device=device)
  
    test_correct = 0
    test_total = 0
    import data
    total_correct=0
    print(len(test_ds))

    counts_r1=np.array([])
    counts_r10=np.array([])
    counts_r30=np.array([])
    counts_r50=np.array([])
    counts_r60=np.array([])
    counts_r70=np.array([])
    counts_r80=np.array([])
    counts_r90=np.array([])
    counts_r100=np.array([])
    counts_r200=np.array([])
    counts_r300=np.array([])
    counts_r400=np.array([])
    counts_r500=np.array([])
    counts_r600=np.array([])
  
    topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]
    counts_rs = {}
    for k in topk1:
                counts_rs[f'counts_r{k}'] = np.array([])
    if split=='train':
        embedding_folder=f'parameters/image/lora/trunks_train/embeddings_{vision_num_blocks}_train.pth'
    elif split=='val':
        embedding_folder=f'parameters/image/lora/trunks_val/embeddings_{vision_num_blocks}.pth'
    else:
        embedding_folder=f'parameters/image/lora/trunks/embeddings_{vision_num_blocks}.pth'
    with torch.no_grad():
        checkpoint = torch.load(embedding_folder)
        # 获取模型参数和张量
        embeddings_vision= checkpoint['audio_embeddings']
        print(len(embeddings_vision))
    with torch.no_grad():
        inputs = {
                ModalityType.TEXT: data.load_and_transform_text(test_ds.text_list, device),
            }
        
        embeddings_text = model_1(inputs)[ModalityType.TEXT]
        for batch_idx, (x, target) in enumerate(test_dl2):
            
            x = x.to(device)
            target=[t.to(device) for t in target]
            try:
                 match_value_1 = embeddings_vision[batch_idx*batch_size:batch_idx*batch_size+batch_size]@embeddings_text.T 
            # embeddings = model_1(inputs)
            # match_value_1 = embeddings[ModalityType.VISION]@embeddings[ModalityType.TEXT].T 
            except:
                 match_value_1 = embeddings_vision[batch_idx*batch_size:]@embeddings_text.T 
                 print(match_value_1.shape)      
            result_1 = torch.softmax(match_value_1, dim=-1)
            # k_values = [10, 30, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]
            # topk_indices_list = []

            # for k in k_values:
            #     _, topk_indices = torch.topk(result_1, k=k, dim=-1)
            #     topk_indices_list.append(topk_indices)

            _, predicted = torch.max(result_1, -1)
            _, topk_indices = torch.topk(result_1, k=10, dim=-1)
            _, topk_indices_30 = torch.topk(result_1, k=30, dim=-1)
            _, topk_indices_50 = torch.topk(result_1, k=50, dim=-1)
            _, topk_indices_60 = torch.topk(result_1, k=60, dim=-1)
            _, topk_indices_70 = torch.topk(result_1, k=70, dim=-1)
            _, topk_indices_80 = torch.topk(result_1, k=80, dim=-1)
            _, topk_indices_90 = torch.topk(result_1, k=90, dim=-1)
            _, topk_indices_100 = torch.topk(result_1, k=100, dim=-1)
            _, topk_indices_200 = torch.topk(result_1, k=200, dim=-1)
            _, topk_indices_300 = torch.topk(result_1, k=300, dim=-1)
            _, topk_indices_400 = torch.topk(result_1, k=400, dim=-1)
            _, topk_indices_500 = torch.topk(result_1, k=500, dim=-1)
            _, topk_indices_600 = torch.topk(result_1, k=600, dim=-1)
            try:
                counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
                counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r30=np.concatenate([counts_r30, [int(any(topk_indices_30[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r50=np.concatenate([counts_r50, [int(any(topk_indices_50[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r60=np.concatenate([counts_r60, [int(any(topk_indices_60[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r70=np.concatenate([counts_r70, [int(any(topk_indices_70[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r80=np.concatenate([counts_r80, [int(any(topk_indices_80[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r90=np.concatenate([counts_r90, [int(any(topk_indices_90[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                
                counts_r100=np.concatenate([counts_r100, [int(any(topk_indices_100[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r200=np.concatenate([counts_r200, [int(any(topk_indices_200[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r300=np.concatenate([counts_r300, [int(any(topk_indices_300[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r400=np.concatenate([counts_r400, [int(any(topk_indices_400[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r500=np.concatenate([counts_r500, [int(any(topk_indices_500[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                counts_r600=np.concatenate([counts_r600, [int(any(topk_indices_600[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            except:
                print(len(predicted))
                print(len(target))
            logging.info(f"batch_idx = {batch_idx} ")
           
        np.savetxt(f'./results/imagenet/lora_{split}/R1/v{v_block}_t{t_block}.txt',counts_r1,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R10/v{v_block}_t{t_block}.txt',counts_r10,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R30/v{v_block}_t{t_block}.txt',counts_r30,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R50/v{v_block}_t{t_block}.txt',counts_r50,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R60/v{v_block}_t{t_block}.txt',counts_r60,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R70/v{v_block}_t{t_block}.txt',counts_r70,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R80/v{v_block}_t{t_block}.txt',counts_r80,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R90/v{v_block}_t{t_block}.txt',counts_r90,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R100/v{v_block}_t{t_block}.txt',counts_r100,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R200/v{v_block}_t{t_block}.txt',counts_r200,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R300/v{v_block}_t{t_block}.txt',counts_r300,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R400/v{v_block}_t{t_block}.txt',counts_r400,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R500/v{v_block}_t{t_block}.txt',counts_r500,fmt='%d')
        np.savetxt(f'./results/imagenet/lora_{split}/R600/v{v_block}_t{t_block}.txt',counts_r600,fmt='%d')
        # for k, counts_array in counts.items():
        #     np.savetxt(f'./results/imagenet/lora_train/R{k}/v{v_block}_t{t_block}.txt', counts_array, fmt='%d')

        #np.savetxt(f'./results/imagenet/5000/R1/v{v_block}_t{t_block}.txt',counts_r1,fmt='%d')
           
    #total_acc = test_acc.compute()

        
def main():
    run_inference()
    
if __name__ == "__main__":
    main()