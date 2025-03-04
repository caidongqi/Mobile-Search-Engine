import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import csv
import torch
import torch.nn as nn
import torchaudio
import json
from tqdm import tqdm
class CoCo_t2i_Dataset(Dataset):
    def __init__(self,caption_path:str=not None,split:str=not None,images_dir:str=not None,):        
        imagepath_caption_list=[]
        caption_list=[]
        image_ids={}
        image_count=0
        pre_deal_file_path=f'coco_{split}_text_imagepath_pairs.list'
        if not os.path.exists(pre_deal_file_path):
            info=json.load(open(caption_path,mode='r'))
            #info=self.data
            images=info['images']
            anotations=info['annotations']
            for ano in tqdm(anotations):
                image_id=ano['image_id']
                caption=ano['caption']
                caption_list.append(caption)
                for image in images:
                    id=image['id']
                    file_name=image['file_name']
                    if id==image_id:
                        image_path=os.path.join(images_dir,file_name)
                        if image_path not in image_ids:
                            image_ids[image_path]=image_count
                            image_count+=1
                        imagepath_caption_list.append((caption,image_path))
                        break
            torch.save(imagepath_caption_list,pre_deal_file_path)
        else:
            imagepath_caption_list=torch.load(pre_deal_file_path)
        
        image_item_list={}
        image_item_file_path=f'coco_{split}_image_item_pairs.list'
        if not os.path.exists(image_item_file_path):
            dataset = os.listdir(images_dir)
            self.datadir=sorted(dataset)
            item=0
            for image in self.datadir:
                image_path=os.path.join(images_dir,image)
                image_item_list[image_path]=item
                item+=1
            torch.save(image_item_list,image_item_file_path)
        else:
            image_item_list=torch.load(image_item_file_path)
        
        self.image_item_list=image_item_list
        self.captions=[]
        self.images_paths=[]
        for caption,image in imagepath_caption_list:
            self.captions.append(caption)
            self.images_paths.append(image)

    def __len__(self):
        # 返回数据集的长度
        return len(self.captions)

    def __getitem__(self, idx):
        text=self.captions[idx]
        image=self.images_paths[idx]
        target=self.image_item_list[image]
        return text,target
      

        

# # # # 使用示例
# CoCo_dataset = CoCo_t2i_Dataset(split="val",images_dir='/home/share/pc/COCO/val2017',caption_path='/home/share/pc/COCO/captions_val2017.json')
# test_dl = DataLoader(dataset=CoCo_dataset, batch_size=10, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)

# with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(test_dl):
#             print(x)
#             print(target)
#             #x=x.to(device)
#             target = target.to(device)
          