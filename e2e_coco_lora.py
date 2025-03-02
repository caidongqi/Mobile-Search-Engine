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
parameter_embedding_folder=f'parameters/image/coco' # e2e still use val
#lora path
lora_dir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/progressive/e20/32"
#predicter model path
model_parameter=f'parameters/image/coco/model/image_S={S}_train_10_lora_bs128_zero.pth'
# f'parameters/image/coco/model/image_S={S}_train_10_lora_bs128_zero.pth' for zero
# f'parameters/image/coco/model/image_S={S}_train_10_lora_bs128.pth' for 32


coarse_embedding_path = f'{parameter_embedding_folder}/embeddings_{N}_trunk_lora.pth' # TODO: currently, those embeddings are computed by models without lora tuning
# coarse_embedding_path = f"/home/u2021010261/share/cdq/Mobile-Search-Engine/parameters/image/coco/trunk/embeddings_{N}_true.pth"
coarse_embedding_path = f"parameters/image/coco/embeddings_{N}_true.pth"
fine_model_embeddings = f'{parameter_embedding_folder}/embeddings_{full_layer}_trunk_lora.pth'
text_embeddings_dir = f'{parameter_embedding_folder}/text_embeddings_trunk_lora.pt'

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
test_ds = CoCo_t2i_Dataset(json_file=coco_annotation_file,datadir=coco_datadir,device=device)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)


 
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


# Step 2. Dynamic embedding: 每个数据动态存储 m 层
# Load predict model
input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 32  # 根据层数的范围确定输出层大小
predict_model = MyModel(input_size, output_size)  # 请确保input_size和output_size已定义
predict_model.to(device)
# 加载已保存的模型参数
predict_model.load_state_dict(torch.load(model_parameter,map_location=device))
mean=0

if os.path.exists(layers_path):
    # 打开数据文件
    with open(layers_path, 'rb') as f:
        layers = pickle.load(f)
    logging.info('步骤2--layer已加载')
    sum=0
    for i in range(len(layers)):
        sum+=layers[i]
    mean=sum/len(layers)
    logging.info(f"Dynamic embedding layer的平均值为:{mean}")
else:
    layers=[]
    
    for embedding_item in coarse_embeddings[ModalityType.VISION]:       
            embedding_item=embedding_item.to(device)
            layer=predict_model(embedding_item.float())
            _, layer1 = torch.max(layer, 0)
            # logging.info(layer1)
            layers.append(layer1+1)
    # 保存数据到文件
    sum=0
    for i in range(len(layers)):
        sum+=layers[i]
    mean=sum/len(layers)
    logging.info(f"Dynamic embedding layer的平均值为:{mean}")
    with open(layers_path, 'wb') as f:
        pickle.dump(layers, f)

    logging.info('步骤2--layer已保存')

if os.path.exists(coarse_embedding_dynamic_path):
    coarse_embedding_dynamic={}
    with torch.no_grad():
        checkpoint = torch.load(coarse_embedding_dynamic_path, map_location=torch.device(args.device))
        # 获取模型参数和张量
        coarse_embedding_dynamic[ModalityType.VISION]= checkpoint['vision_embeddings']
    logging.info('步骤2--dynamic存在,已加载')
else:
    coarse_embedding_dynamic={}
    with torch.no_grad():
        for i in range(len(layers)):
            current_coarse_embedding_dynamic_path=f'{parameter_embedding_folder}/embeddings_{layers[i]}_trunk_lora.pth'
            if os.path.exists(current_coarse_embedding_dynamic_path):
                current_embeddings = torch.load(current_coarse_embedding_dynamic_path, map_location=torch.device(args.device))['vision_embeddings'][i]
                if coarse_embedding_dynamic:
                    coarse_embedding_dynamic[ModalityType.VISION] = torch.cat([coarse_embedding_dynamic[ModalityType.VISION], current_embeddings.unsqueeze(0).to(coarse_embedding_dynamic[ModalityType.VISION].device)], dim=0)
                else:
                    coarse_embedding_dynamic[ModalityType.VISION] = current_embeddings.unsqueeze(0)
                del current_embeddings
                
            else:
                logging.info(f"no {current_coarse_embedding_dynamic_path}")
                logging.info('please get embeddings first')
                exit(0)
                    
        torch.save({
                'vision_embeddings': coarse_embedding_dynamic[ModalityType.VISION]
            }, coarse_embedding_dynamic_path)
        logging.info('步骤2--dynamic不存在,已保存')


# Step.3 根据query进行match到前K个数据
fine_model = imagebind_model.imagebind_huge(pretrained=True)
if lora_dir != None:
    fine_model.modality_trunks.update(LoRA.apply_lora_modality_trunks(fine_model.modality_trunks, rank=4,
                                                                    layer_idxs={
                                                                                ModalityType.VISION: [i for i in range(1,full_layer)]},
                                                                                modality_names=[ ModalityType.VISION]))
    LoRA.load_lora_modality_trunks(fine_model.modality_trunks, checkpoint_dir=lora_dir, postfix = "_last")
    load_module(fine_model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir)
    load_module(fine_model.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir)
fine_model=fine_model.to(device)
fine_model.eval()


K_list=[1, 2, 5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,300,400,500,600] # top k list
K_caption_correct_list = {} #  correct/not list for all test images with different K, e.g., {"K=1": [1,1,0,0...], 'K=5":[...], ...}
shortlist={} # store concrete path, text label
shortlist_item={} # the index of label

for k in K_list:
    K_caption_correct_list[f'K={k}'] = np.array([])
    shortlist[f'K={k}']=[]
    shortlist_item[f'K={k}']=[]

if not os.path.exists(text_embeddings_dir):
    logging.info('text_embeddings不存在,开始生成')
    all_text_embeddings = []
    with torch.no_grad():
        # dynamic embedding classification
        for batch_idx, (x, target) in enumerate(test_dl):
            target = target.to(device)
            
            inputs = {ModalityType.TEXT: data.load_and_transform_text(x, device)}

            text_embeddings = fine_model(inputs)[ModalityType.TEXT]
            all_text_embeddings.append(text_embeddings)
            logging.info(f"batch_idx = {batch_idx} / {len(test_dl)}")

    # Concatenate all embeddings into a single tensor
    all_embeddings_tensor = torch.cat(all_text_embeddings, dim=0)
    torch.save(all_embeddings_tensor, text_embeddings_dir)
    logging.info(f"Data saved successfully to {text_embeddings_dir}")
    all_text_embedding = all_embeddings_tensor
else:  
    all_text_embeddings = torch.load(text_embeddings_dir, map_location=torch.device(args.device))
    logging.info('text_embeddings存在,已加载')

if os.path.exists(shortlist_path):
    with open(shortlist_path, 'rb') as f:
        # shortlist = pickle.load(f)
        shortlist_item = pickle.load(f)
        #print('shortlist_item',shortlist_item)
    logging.info('步骤3--shortlist存在,已加载')

    show_dynamic_embedding_accuracy = False
    if show_dynamic_embedding_accuracy:
        # just for computing the dynamic embedding accuracy
        with torch.no_grad():
            
            # dynamic embedding classification
            for batch_idx, (x, target) in enumerate(test_dl):
                target = target.to(device)
                #target=imagebind_targets[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
                if batch_idx==len(test_dl)-1:
                    text_embeddings = all_text_embeddings[batch_idx*batch_size:]
                else:
                    text_embeddings = all_text_embeddings[batch_idx*batch_size:(batch_idx+1)*batch_size]

                match_value = text_embeddings@coarse_embedding_dynamic[ModalityType.VISION].T 
        
                result = torch.softmax(match_value, dim=-1)
                _, predicted = torch.max(result, dim=-1)
                top_indices_list = [torch.topk(result, k=k, dim=-1)[1] for k in K_list]
                
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
else:    
    # # just for computing the dynamic embedding accuracy and store top_indices_list for Step 4 final r@k accuracy
    logging.info('步骤3--shortlist不存在,开始保存')
    with torch.no_grad():
        
        # dynamic embedding classification
        for batch_idx, (x, target) in enumerate(test_dl):
            target = target.to(device)
            #target=imagebind_targets[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
            if batch_idx==len(test_dl)-1:
                text_embeddings = all_text_embeddings[batch_idx*batch_size:]
            else:
                text_embeddings = all_text_embeddings[batch_idx*batch_size:(batch_idx+1)*batch_size]

            match_value = text_embeddings@coarse_embedding_dynamic[ModalityType.VISION].T 
    
            result = torch.softmax(match_value, dim=-1)
            _, predicted = torch.max(result, dim=-1)
            top_indices_list = [torch.topk(result, k=k, dim=-1)[1] for k in K_list]
            
            for k, top_indices, item in zip(K_list, top_indices_list, K_caption_correct_list):
                if k == 1:
                    K_caption_correct_list[item] = np.concatenate([K_caption_correct_list[item], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]]) # np.concatenate([current_total], [per_batch])
                else:
                    K_caption_correct_list[item] = np.concatenate([K_caption_correct_list[item], [int(any(top_indices[i] == target[i].to(predicted.device))) for i in range(len(target))]])
                #save top_indices_list into shortlist file
                for i,row in enumerate(top_indices):
                    list_value=[] # the retrieved figure coarse embedding value
                    list_item=[] # the retrieved figure index
                    for item_r in row:
                        list_value.append(coarse_embedding_dynamic['vision'][item_r.item()])
                        list_item.append(item_r.item())
                    # shortlist[item].append(list_value)
                    shortlist_item[item].append(list_item)

                tested_caption_length = len(K_caption_correct_list['K=1'])
                r1=(np.sum(K_caption_correct_list['K=1']))/tested_caption_length 
                r5=(np.sum(K_caption_correct_list['K=5']))/tested_caption_length
                r10=(np.sum(K_caption_correct_list['K=10']))/tested_caption_length
                
            # 保存 shortlist 和 shortlist_item 到本地文件
            with open(shortlist_path, 'wb') as f:
                # pickle.dump(shortlist, f)
                pickle.dump(shortlist_item, f)

            logging.info(f'batch_idx: {batch_idx}, dynamic embedding的准确率:{r1}_{r5}_{r10}')

    logging.info(f"Data saved successfully to {shortlist_path}")

    # for future final_results.csv saving
    results_dynamic=[]
    lists=[]
    results_dynamic.append('dynamic')
    results_dynamic.append(N)
    results_dynamic.append(Q)
    results_dynamic.append(S)
    results_dynamic.append(mean)
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
        

# Step.4 再次进行fine-grained embedding # TODO: 现在是对image fine grained，然后对top k的text list进行match；未来应该是top k的image list进行match
batch_size=1

test_dl_final = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

K_caption_correct_list_final = {} # this is different from dynamic embedding because all images are fullly embedded
for k in K_list:
    if k<=Q:
        K_caption_correct_list_final[f'K={k}'] = np.array([])
embeddings={}
embeddings_all={}

with torch.no_grad():
    checkpoint = torch.load(fine_model_embeddings, map_location=torch.device(args.device))
    # 获取模型参数和张量
    fine_image_embeddings_value= checkpoint['vision_embeddings']
    logging.info(f"step 4: fine-grained embedding")

    for batch_idx, (x, target) in enumerate(test_dl_final):
        target = target.to(device)
        #target=imagebind_targets[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
        text_embeddings = all_text_embeddings[batch_idx*batch_size]

        # print('batch_idx',batch_idx)
        # print('target',target)
        # print(shortlist_item[f'K={Q}'][batch_idx*batch_size])
        item = shortlist_item[f'K={Q}'][batch_idx*batch_size] # Batch_size = 1
        image_embeddings = fine_image_embeddings_value[item]
        
        match_value = text_embeddings@image_embeddings.T 
        result = torch.softmax(match_value, dim=-1)
        _, predicted = torch.max(result, dim=-1)
        _, topk_indices = torch.topk(result, k=Q, dim=-1)
        
        top_indices_list = [torch.topk(result, k=k, dim=-1)[1] if k <= Q else None for k in K_list]

        predicted=torch.Tensor([shortlist_item[f'K={Q}'][batch_idx*batch_size][predicted]])
        
        for k, top_indices, item in zip(K_list, top_indices_list, K_caption_correct_list_final):
            if k == 1:
                K_caption_correct_list_final[item] = np.concatenate([K_caption_correct_list_final[item], [int(predicted[i] == target[i].to(predicted.device)) for i in range(predicted.numel())]]) # np.concatenate([current_total], [per_batch])
            elif k<=Q:            
                results_k=[]
                for i_k in range(k):
                    results_k.append(shortlist_item[f'K={Q}'][batch_idx][top_indices[i_k].item()])
                t=target[0]
                bool_list=[int(results_k[i] == t.item()) for i in range(len(results_k))]
                flag = 1 if any(bool_list) else 0
                # print(flag)
                # print(K_image_correct_list_final[item])
                K_caption_correct_list_final[item] = np.concatenate([K_caption_correct_list_final[item],[flag]])
                # exit(0)
            else :
                break

        tested_caption_length = len(K_caption_correct_list_final['K=1'])
        r1=(np.sum(K_caption_correct_list_final['K=1']))/tested_caption_length 
        # r5=(np.sum(K_caption_correct_list_final['K=5']))/tested_caption_length
        # r10=(np.sum(K_caption_correct_list_final['K=10']))/tested_caption_length
        if batch_idx%100==0:
            logging.info(f"batch_idx = {batch_idx}, r1={r1}, test_total = {tested_caption_length}")
        # logging.info(f"fine-grained embedding : {r1}_{r5}_{r10}")
        
results=[]
lists=[]
results.append('total') 
results.append(N)
results.append(Q)
results.append(S)
results.append(mean)
for counts in K_caption_correct_list_final:
    correct=np.sum(K_caption_correct_list_final[counts])/tested_caption_length
    results.append(str(correct))

# 数据
data1 = [
    results
]

# # 指定CSV文件路径
csv_file_path = f'e2e_coco_lora_{version}.csv'

with open(csv_file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入新数据
    for row in data1:
        writer.writerow(row)
        
        

