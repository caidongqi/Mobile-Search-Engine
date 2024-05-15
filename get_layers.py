import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO, force=True)
import os
csv_file_path = "/home/u2021010261/data/cdq/clotho/clotho_captions_evaluation.csv"
data_dir="/home/u2021010261/data/cdq/clotho/evaluation"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=["/home/u2021010261/data/cdq/clotho/evaluation"+file for file in audio_list]
import random
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def find_first_nonzero_row(array, column_index):
    for i, row in enumerate(array):
        if row[column_index] != 0:
            return i
    return 13

filenames = ['t24_a{0}.txt'.format(i) for i in range(1,13)]
files = [os.path.join('./results/clotho/lora/text_nohead/R10',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]
audio_layers={}
for i in range(len(audio_list)):
    audio_layers[i]=[]
for i in range(len(audio_list)):
    audio_id=i
    for j in range(5*audio_id,5*audio_id+5):        
        audio_layers[i].append(find_first_nonzero_row(np.array(counts),j))
audio_max=[]
for i in range(len(audio_list)):
    audio_id=i
    audio_max.append(min(audio_layers[i]))
for i in range(len(audio_list)):
    if audio_max[i]==13:
        audio_max[i]=0
np.savetxt(f'./results/clotho/lora/text_nohead/R10/layers_min_all.txt',audio_max,fmt='%d')
    
        