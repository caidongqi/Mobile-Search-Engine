import logging
import torch
import data
import pickle
from models import imagebind_model
from models.mlp import MyModel
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from api.coco import CoCoDataset
from api.coco_text2image import CoCo_t2i_Dataset
import os
import argparse
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--N", type=int, default=2, help="First get N embeddings")
parser.add_argument("--Q", default=100,type=int, help="Fine grained embedding scope after query")
parser.add_argument("--S", default=10,type=int, help="Grain for predict model, larger S, smaller average predicted layer")
parser.add_argument("--split", default='val',type=str, help="train or val")
parser.add_argument("--device", default='cuda:0',type=str, help="gpu device id (if applicable)")
parser.add_argument("--version", default='00',type=str, help="gpu device id (if applicable)")

args = parser.parse_args()
N=args.N
Q=args.Q
S=args.S
split=args.split
version=args.version
# N=2
# Q=100
logging.info(f"N={N},Q={Q}")
full_layer=32 #audio:12 image:32
device = args.device if torch.cuda.is_available() else "cpu"


# testset path
coco_annotation_file = "/home/u2021010261/share/pc/COCO/captions_val2017.json"
coco_datadir="/home/u2021010261/share/pc/COCO/val2017"
#embedding path
parameter_embedding_folder=f'parameters/image/coco/val' # e2e still use val
#lora path
lora_dir=f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/trunk/ratio3/e12/32'

coarse_embedding_path = f'{parameter_embedding_folder}/embeddings_{N}_trunk_lora.pth' # TODO: currently, those embeddings are computed by models without lora tuning
# coarse_embedding_path = f"/home/u2021010261/share/cdq/Mobile-Search-Engine/parameters/image/coco/trunk/embeddings_{N}_true.pth"
coarse_embedding_path = f"parameters/image/coco/embeddings_{N}_true.pth"
fine_model_embeddings = f'{parameter_embedding_folder}/embeddings_{full_layer}_trunk_lora.pth'
text_embeddings_dir = f'{parameter_embedding_folder}/text_embeddings_trunk_lora_v2.pt'

# 下面的三个名字，跑的时候尽量改一下
# dynamic embeddings
coarse_embedding_dynamic_path=f'{parameter_embedding_folder}/dynamic/N={N}_S={S}_v{version}.pth'

#save layers
layers_path=f"{parameter_embedding_folder}/layers/N={N}_S={S}_v{version}.pkl"
shortlist_path=f"{parameter_embedding_folder}/shortlist/shortlist_data_N={N}_S={S}_v{version}.pkl" # Different Q could share the same shortlist

#imagebind target 
imagebind_target_path="parameters/imagebind_targets/imagebind_32.pt"
imagebind_targets=torch.load(imagebind_target_path)

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

batch_size = 64
CoCo_dataset = CoCo_t2i_Dataset(split="val",images_dir='/home/share/pc/COCO/val2017',caption_path='/home/share/pc/COCO/captions_val2017.json')
test_dl = DataLoader(dataset=CoCo_dataset, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
test_dl1 = DataLoader(dataset=CoCo_dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)


model = imagebind_model.imagebind_huge(pretrained=True,vision_num_blocks=32)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)


# Load fine-tuned text heads
load_module(model.modality_heads, module_name="heads",
            checkpoint_dir=lora_dir, device =device)


#
# model = DataParallel(model)
# model=model.cuda()
model.to(device)
model.eval()

# Step 1: 存储N层
coarse_embeddings={} # load from coarse_embedding_path
if os.path.exists(coarse_embedding_path):
    with torch.no_grad():
        checkpoint = torch.load(coarse_embedding_path)
        # 获取模型参数和张量
        coarse_embeddings[ModalityType.VISION]= checkpoint['vision_embeddings']
        logging.info('步骤1已加载')
else : 
    logging.info('步骤1未加载')
    logging.info(f'没有参数{coarse_embedding_path}')
    exit(0)


ground_truth='results/coco_lora_val/R10/layers_min.txt'
layers=np.loadtxt(ground_truth)
print(np.mean(layers))
coarse_embedding_dynamic={}
text_embeddings={}
with torch.no_grad():
    # if not os.path.exists(coarse_embedding_dynamic_path):
        for i in range(len(layers)):
            current_coarse_embedding_dynamic_path=f'{parameter_embedding_folder}/embeddings_{int(layers[i]+1)}_trunk_lora.pth'
            if os.path.exists(current_coarse_embedding_dynamic_path):
                current_embeddings = torch.load(current_coarse_embedding_dynamic_path, map_location=torch.device(args.device))['vision_embeddings'][i]
                if coarse_embedding_dynamic:
                    coarse_embedding_dynamic[ModalityType.VISION] = torch.cat([coarse_embedding_dynamic[ModalityType.VISION], current_embeddings.unsqueeze(0).to(coarse_embedding_dynamic[ModalityType.VISION].device)], dim=0)
                else:
                    coarse_embedding_dynamic[ModalityType.VISION] = current_embeddings.unsqueeze(0)
                del current_embeddings
                    
        torch.save({
                'vision_embeddings': coarse_embedding_dynamic[ModalityType.VISION]
            }, coarse_embedding_dynamic_path)
        logging.info('步骤2--dynamic不存在,已保存')

    # if not os.path.exists(text_embeddings_dir):
        # for i in range(len(CoCo_dataset)):
        for batch_idx, (x, target) in enumerate(test_dl1):
            current_text_embedding_dynamic_path=f'{parameter_embedding_folder}/text_embeddings_{int(layers[target]+1)}_trunk_lora.pth'
            if os.path.exists(current_text_embedding_dynamic_path):
                current_embeddings = torch.load(current_text_embedding_dynamic_path, map_location=torch.device(args.device))['text_embeddings'][batch_idx]
                if text_embeddings:
                    text_embeddings[ModalityType.TEXT] = torch.cat([text_embeddings[ModalityType.TEXT], current_embeddings.unsqueeze(0).to(text_embeddings[ModalityType.TEXT].device)], dim=0)
                else:
                    text_embeddings[ModalityType.TEXT] = current_embeddings.unsqueeze(0)
                del current_embeddings
                    
        torch.save({
                'text_embeddings': text_embeddings[ModalityType.TEXT]
            }, text_embeddings_dir)
        logging.info('步骤2--dynamic不存在,已保存')

K_list=[1, 2, 5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600] # top k list
K_caption_correct_list = {} #  correct/not list for all test images with different K, e.g., {"K=1": [1,1,0,0...], 'K=5":[...], ...}
shortlist={} # store concrete path, text label
shortlist_item={} # the index of label

for k in K_list:
    K_caption_correct_list[f'K={k}'] = np.array([])
    shortlist[f'K={k}']=[]
    shortlist_item[f'K={k}']=[]


all_text_embeddings = torch.load(text_embeddings_dir, map_location=torch.device(args.device))
coarse_embedding_dynamic=torch.load(coarse_embedding_dynamic_path, map_location=torch.device(args.device))
logging.info('text_embeddings存在,已加载')

if os.path.exists(shortlist_path):
    with open(shortlist_path, 'rb') as f:
        shortlist_item = pickle.load(f)
    logging.info('步骤3--shortlist存在,已加载')
    show_dynamic_embedding_accuracy = True
    if show_dynamic_embedding_accuracy:
        # just for computing the dynamic embedding accuracy
        correct_layer=[]
        with torch.no_grad():           
            # dynamic embedding classification
            for batch_idx, (x, target) in enumerate(test_dl):
                target = target.to(device)
                if batch_idx==len(test_dl)-1:
                    text_embeddings = all_text_embeddings['text_embeddings'][batch_idx*batch_size:]
                else:
                    text_embeddings = all_text_embeddings['text_embeddings'][batch_idx*batch_size:(batch_idx+1)*batch_size]

                match_value = text_embeddings@coarse_embedding_dynamic['vision_embeddings'].T 
                #match_value = embeddings[ModalityType.TEXT].to(coarse_embedding_dynamic[ModalityType.VISION].device)@coarse_embedding_dynamic[ModalityType.VISION].T 
                result = torch.softmax(match_value, dim=-1)
                _, predicted = torch.max(result, dim=-1)
                top_indices_list = [torch.topk(result, k=k, dim=-1)[1] for k in K_list]

            
                for i in range(predicted.numel()):
                    correct_layer.append(layers[int(predicted[i])])


                for k, top_indices, item in zip(K_list, top_indices_list, K_caption_correct_list):
                    if k == 1:

                        K_caption_correct_list[item] = np.concatenate([K_caption_correct_list[item], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]]) # np.concatenate([current_total], [per_batch])
                    else:
                        K_caption_correct_list[item] = np.concatenate([K_caption_correct_list[item], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])

                tested_caption_length = len(K_caption_correct_list['K=1'])
                r1=(np.sum(K_caption_correct_list['K=1']))/tested_caption_length 
                r5=(np.sum(K_caption_correct_list['K=5']))/tested_caption_length
                r10=(np.sum(K_caption_correct_list['K=10']))/tested_caption_length
                
                logging.info(f"batch_idx = {batch_idx}, r1={r1},r10={r10}, test_total = {tested_caption_length}")
                
            logging.info(f'dynamic embedding的准确率:{r1}_{r5}_{r10}')
        
        logging.info(f'-------------------------')
                
        file_path = 'array_data.txt'

        # 打开文件，准备写入
        with open(file_path, 'w') as file:
            # 将数组转换为字符串，然后写入文件
            # 这里使用' '作为分隔符，你也可以使用其他分隔符如','或'\t'
            for element in correct_layer:
                file.write(str(element) + '\n')

    # for future final_results.csv saving
    results_dynamic=[]
    lists=[]
    results_dynamic.append('dynamic')
    results_dynamic.append(N)
    results_dynamic.append(Q)
    results_dynamic.append(S)

    for counts in K_caption_correct_list:
        correct=np.sum(K_caption_correct_list[counts] == 1)/len(K_caption_correct_list['K=1']) # TODO: change this to r1 for simpilicity
        results_dynamic.append(str(correct))
        lists.append(str(counts))
    # 数据
    data1 = [
        results_dynamic
    ]

    # # 指定CSV文件路径
    csv_file_path = f'e2e_coco_lora_{version}.csv'

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入新数据
        for row in data1:
            writer.writerow(row)
        

