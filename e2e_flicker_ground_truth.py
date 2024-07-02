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
from api.flickr import flickr8k
import os
import argparse
import csv
import time
# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间
local_time = time.localtime(timestamp)
# 格式化本地时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)



# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--N", type=int, default=2, help="First get N embeddings")
parser.add_argument("--Q", default=20,type=int, help="Fine grained embedding scope after query")
parser.add_argument("--S", default=10,type=int, help="Grain for predict model, larger S, smaller average predicted layer")
parser.add_argument("--split", default='val',type=str, help="train or val")
parser.add_argument("--device", default='cuda:0',type=str, help="gpu device id (if applicable)")
parser.add_argument("--version", default='ground_truth_flicker',type=str, help="gpu device id (if applicable)")

args = parser.parse_args()
N=args.N
Q=args.Q
S=args.S
split=args.split
version=args.version

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'logs/e2e/N={N}_S={S}_{version}_{formatted_time}.log')], 
                    force=True)

# N=2
# Q=100
logging.info(f"N={N},Q={Q}")
full_layer=32 #audio:12 image:32
device = args.device if torch.cuda.is_available() else "cpu"

#embedding path
parameter_embedding_folder=f'parameters/image/flickr8k/val' # e2e still use val
#lora path
lora_dir=f'/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/flickr8k/without_head/trunk/e100/32'

#lora_dir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.checkpoints/lora/coco/trunk/ratio3/e12/32"
#predicter model path
model_parameter=f'parameters/image/coco/model/image_S={S}_val_v1.pth'
coarse_embedding_path = f'{parameter_embedding_folder}/embeddings_{N}.pth' # TODO: currently, those embeddings are computed by models without lora tuning
fine_model_embeddings = f'{parameter_embedding_folder}/embeddings_{full_layer}.pth'
text_embeddings_dir = f'{parameter_embedding_folder}/text_embeddings_N={N}_S={S}_{version}.pt'

# 下面的三个名字，跑的时候尽量改一下
# dynamic embeddings
coarse_embedding_dynamic_path=f'{parameter_embedding_folder}/dynamic/N={N}_S={S}_v{version}.pth'

#save layers
#layers_path=f"/home/u2021010261/share/pc/Mobile-Search-Engine/parameters/image/coco/layers/N={N}_S={S}_v{version}.pkl"
shortlist_path=f"{parameter_embedding_folder}/shortlist/shortlist_data_N={N}_S={S}_v{version}.pkl" # Different Q could share the same shortlist

#imagebind target 
imagebind_target_path="parameters/imagebind_targets/imagebind_32.pt"
imagebind_targets=torch.load(imagebind_target_path)

if not os.path.exists(f'{parameter_embedding_folder}/dynamic'):
    os.makedirs(f'{parameter_embedding_folder}/dynamic', exist_ok=True)

if not os.path.exists(f'{parameter_embedding_folder}/shortlist'):
    os.makedirs(f'{parameter_embedding_folder}/shortlist', exist_ok=True)

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
datadir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.datasets/flickr8k/images"
anne_dir = "/home/u2021010261/data/yx/Mobile-Search-Engine-main/.datasets/flickr8k/captions.txt"
test_ds = flickr8k(root_dir=datadir, anne_dir=anne_dir, split='test')
test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
num_workers=4, pin_memory=True, persistent_workers=True)
test_dl1 = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, drop_last=False,
num_workers=4, pin_memory=True, persistent_workers=True)

text_prompt = 'a photo of {}.'
# 从文件加载字典
with open('/home/u2021010261/data/yx/Mobile-Search-Engine-main/flickr8k_img_dict.pkl', 'rb') as file:
    img_dict = pickle.load(file)

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

ground_truth='results/flickr8k_lora_val_nohead/R10/layers.txt'
layers=np.loadtxt(ground_truth)
logging.info('步骤2--layer已加载')
sum=0
for i in range(len(layers)):
    sum+=layers[i]
mean=sum/len(layers)
logging.info(f"Dynamic embedding layer的平均值为:{mean}")
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
            current_coarse_embedding_dynamic_path=f'{parameter_embedding_folder}/embeddings_{int(layers[i])+1}.pth'
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

# exit(0)

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
    text_embeddings={}
    with torch.no_grad():
        # dynamic embedding classification
        for batch_idx, (_,x, image_name) in enumerate(test_dl1):
            text_x=[text_prompt.format(x)]
            inputs = {ModalityType.TEXT: data.load_and_transform_text(text_x, device)}
            embeddings = fine_model(inputs)
            text_embeddings=embeddings[ModalityType.TEXT]
            all_text_embeddings.append(text_embeddings)
            logging.info(f"batch_idx = {batch_idx} / {len(test_ds)}")

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

    show_dynamic_embedding_accuracy = True
    if show_dynamic_embedding_accuracy:
        # just for computing the dynamic embedding accuracy
        with torch.no_grad():
            # dynamic embedding classification
            for batch_idx, (_, x, image_name) in enumerate(test_dl):
                target = torch.tensor([img_dict[name] for name in image_name]).to(device)
                #text_x = [text_prompt.format(x[i]) for i in range(len(x))]
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
        
        results_dynamic=[]
        lists=[]
        results_dynamic.append('dynamic')
        results_dynamic.append(S)
        results_dynamic.append(N)
        results_dynamic.append(Q)
        
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
        csv_file_path = f'ground_truth_lora_{version}.csv'

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入新数据
            for row in data1:
                writer.writerow(row)
        
else:    
    # # just for computing the dynamic embedding accuracy and store top_indices_list for Step 4 final r@k accuracy
    logging.info('步骤3--shortlist不存在,开始保存')
    with torch.no_grad():
        # dynamic embedding classification
        for batch_idx, (_,x, image_name) in enumerate(test_dl):
            target = torch.tensor([img_dict[name] for name in image_name]).to(device)
            text_x = [text_prompt.format(x[i]) for i in range(len(x))]
            
            #target=imagebind_targets[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
            if batch_idx==len(test_dl)-1:
                text_embeddings = all_text_embeddings[batch_idx*batch_size:]
            else:
                text_embeddings = all_text_embeddings[batch_idx*batch_size:(batch_idx+1)*batch_size]
            # print(text_embeddings)
            # print("------------------")
            # print(coarse_embedding_dynamic[ModalityType.VISION])
            match_value = text_embeddings@coarse_embedding_dynamic[ModalityType.VISION].T 
    
            result = torch.softmax(match_value, dim=-1)
            _, predicted = torch.max(result, dim=-1)
            top_indices_list = [torch.topk(result, k=k, dim=-1)[1] for k in K_list]
            
            for k, top_indices, item in zip(K_list, top_indices_list, K_caption_correct_list):
                if k == 1:
                    K_caption_correct_list[item] = np.concatenate([K_caption_correct_list[item], [int(predicted[i] == target[i].to(predicted.device)) for i in range(target.numel())]]) # np.concatenate([current_total], [per_batch])
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
            
            # pickle.dump(shortlist, f)
            folder_path = os.path.dirname(shortlist_path)
            # 确保文件夹路径存在
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存 shortlist 和 shortlist_item 到本地文件
            with open(shortlist_path, 'wb') as f:
                
                pickle.dump(shortlist_item, f)

            logging.info(f'batch_idx: {batch_idx}, dynamic embedding的准确率:{r1}_{r5}_{r10}')

    logging.info(f"Data saved successfully to {shortlist_path}")

    # for future final_results.csv saving
    results_dynamic=[]
    lists=[]
    results_dynamic.append('dynamic')
    results_dynamic.append(S)
    results_dynamic.append(N)
    results_dynamic.append(Q)
    
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
    csv_file_path = f'ground_truth_lora_{version}.csv'

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入新数据
        for row in data1:
            writer.writerow(row)
        

# Step.4 再次进行fine-grained embedding # TODO: 现在是对image fine grained，然后对top k的text list进行match；未来应该是top k的image list进行match
# batch_size=1
# test_dl_final = DataLoader(dataset=CoCo_dataset, batch_size=1, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)

K_caption_correct_list_final = {} # this is different from dynamic embedding because all images are fullly embedded
for k in K_list:
    if k<=Q:
        K_caption_correct_list_final[f'K={k}'] = np.array([])
embeddings={}
embeddings_all={}
batch_size=1
with torch.no_grad():
    checkpoint = torch.load(fine_model_embeddings, map_location=torch.device(args.device))
    # 获取模型参数和张量
    fine_image_embeddings_value= checkpoint['vision_embeddings']
    logging.info(f"step 4: fine-grained embedding")

    for batch_idx, (_,x, image_name) in enumerate(test_dl1):
        text_embeddings = all_text_embeddings[batch_idx*batch_size]
        target = torch.tensor([img_dict[name] for name in image_name]).to(device)
        text_x = [text_prompt.format(x[i]) for i in range(len(x))]
            
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
results.append(S)
results.append(N)
results.append(Q)

results.append(mean)
for counts in K_caption_correct_list_final:
    correct=np.sum(K_caption_correct_list_final[counts])/tested_caption_length
    results.append(str(correct))

# 数据
data1 = [
    results
]

# # 指定CSV文件路径
csv_file_path = f'e2e_lora_{version}.csv'

with open(csv_file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入新数据
    for row in data1:
        writer.writerow(row)
        
        

