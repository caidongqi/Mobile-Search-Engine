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
#device="cpu"
#device = "cuda:6" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True
imagenet_datadir = "/data/yx/ImageBind/.datasets/imagenet"
#imagenet_datadir = "/data/air/pc/Mobile-Search-Engine/.datasets/one_imagenet"
coco_datadir="/data/air/pc/ImageBind/dataset/val2017/val2017"
test_coco="/data/air/pc/ImageBind/dataset/tempoimage"
datadir1='/data/air/pc/Mobile-Search-Engine/datasets/imagenet-10'
datadir2="/data/air/pc/i-Code/i-Code-V3/dataset/tempo-10"
coco_annotation_file='/data/air/pc/ImageBind/dataset/annotations_trainval2017/annotations/instances_val2017.json'
lora_dir = '/data/air/pc/Mobile-Search-Engine/.checkpoints/550_epochs_lora'

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")

# 添加命令行参数
#parser.add_argument("audio_num_blocks", type=int, help="Number of audio blocks")

# parser.add_argument("--audio_num_blocks", default=12, type=int, help="Number of audio blocks")
parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--vision_num_blocks", default=26,type=int, help="Number of audio blocks")
# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
vision_num_blocks=args.vision_num_blocks

vision_num_blocks_1=vision_num_blocks
audio_num_blocks_2=12
device_ids = [4,5,6] 
device = "cuda:6" if torch.cuda.is_available() else "cpu"

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_1 = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=vision_num_blocks_1)
#model_2 = imagebind_model.imagebind_huge(pretrained=True,audio_num_blocks=audio_num_blocks_2)
v_block=len(model_1.modality_trunks["vision"].blocks)
t_block=len(model_1.modality_trunks["text"].blocks)
a_block=len(model_1.modality_trunks["audio"].blocks)
i_block=len(model_1.modality_trunks["imu"].blocks)
model_1=model_1.cuda()
model_1 = model_1.to(device_ids[0]) 
model_1 = DataParallel(model_1,device_ids=device_ids)

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
    test_ds = ImageNetDataset(datadir=imagenet_datadir, split="val", transform=data_transform)
    #test_dl1=DataLoader(dataset=test_ds1, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
    # 取前5000个数据
    num_samples = 5000
    test_dl2 = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

    test_dl2 = itertools.islice(test_dl, num_samples)
    import os
    test_acc = Accuracy(task="multiclass", num_classes=1000, average="micro",device=device)
    file_count = sum(1 for entry in os.scandir(coco_datadir) if entry.is_file())
    test_correct = 0
    test_total = 0
    import data
    total_correct=0
    print(len(test_ds))
    counts_r1=np.array([])
    counts_r10=np.array([])
   
  
    topk1=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]
    counts_rs = {}
    for k in topk1:
                counts_rs[f'counts_r{k}'] = np.array([])
    with torch.no_grad():
        for batch_idx, (x, target,imgs) in enumerate(test_dl2):
            
            x = x.to(device)
            target=[t.to(device) for t in target]
            
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(test_ds.text_list, device),
            }
            embeddings = model_1(inputs)
            match_value_1 = embeddings[ModalityType.VISION]@embeddings[ModalityType.TEXT].T 
            print(match_value_1.shape)
            # num=5
            # topk_indices = torch.topk(match_value_1, num, dim=-1).indices
            # # 获取每行最大值的索引
            # max_indices = torch.argmax(match_value_1, dim=1)
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            _, topk_indices = torch.topk(result_1, k=10, dim=-1)
            #topk=[5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600]
            counts_r1 = np.concatenate([counts_r1, [int(predicted[i] == target[i].to(predicted.device)) for i in range(len(predicted))]])
            # #counts_r1 = np.concatenate([counts_r1, [any(predicted[i] == target[i]) for i in range(len(predicted))]])
            # #topk_indices=topk_indices.T
            counts_r10=np.concatenate([counts_r10, [int(any(topk_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
            
            logging.info(f"batch_idx = {batch_idx} ")
            # if(np.sum(counts_rs['counts_r1'] == 1)):
            #     print(f"batch_idx不为0:{imgs}")
            #     embed=embeddings[ModalityType.VISION]
            #     print(embed)
            #     torch.save(embed, "output_tensor.pt")
            #     return imgs
            # for i in range(top_k_indices.size(0)):
            #     row_tensor1 = top_k_indices[i, :]
            #     row_tensor1= torch.add(row_tensor1, 1)
            #     # 提取每个子tensor的第i列
            #     column_values = [tensor.item() for tensor in target]
            #     extracted_values = [str(item) for item in column_values]

            #     # 组成新的tensor
            #     new_tensor = torch.cat([torch.tensor(extracted_values)])
            #     new_tensor=torch.unique(new_tensor.to(device))
            #     intersection = torch.unique(torch.cat((row_tensor1, new_tensor)))
            #     #intersection = torch.intersection(row_tensor1, new_tensor)
            #     if len(intersection)<len(row_tensor1) + len(new_tensor):
            #             correct+=1
        np.savetxt(f'./results/imagenet/5000/R10/v{v_block}_t{t_block}.txt',counts_r10,fmt='%d')
        np.savetxt(f'./results/imagenet/5000/R1/v{v_block}_t{t_block}.txt',counts_r1,fmt='%d')
           
    #total_acc = test_acc.compute()
    
        
def main():
    run_inference()
    
if __name__ == "__main__":
    main()