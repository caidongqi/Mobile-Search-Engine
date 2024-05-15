# 2.每个数据动态存储 m 层
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
import data
import torch.nn as nn
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
import pandas as pd

logging.basicConfig(level=logging.INFO, force=True)
import os
csv_file_path = "/home/u2021010261/data/cdq/clotho/clotho_captions_evaluation.csv"
data_dir="/home/u2021010261/data/cdq/clotho/evaluation/"
f_s=os.listdir(data_dir)
print(len(f_s))
pf=pd.read_csv(csv_file_path,sep=',') # 假设数据集以CSV文件形式提供
text_list = pf[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.flatten().tolist()
audio_list=pf[['file_name']].values.flatten().tolist()
audio_path=[data_dir+file for file in audio_list]
embeddings={}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_ids = [0,1,2] 
N=1
with torch.no_grad():
        checkpoint = torch.load(f'parameters/audio/trunks+post/embeddings_{N}.pth', map_location='cuda:0')
        # 获取模型参数和张量
        embeddings[ModalityType.AUDIO]= checkpoint['audio_embeddings']
        print(1)
# 定义模型结构
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型实例
input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 12  # 根据层数的范围确定输出层大小
predict_model = MyModel(input_size, output_size)  # 请确保input_size和output_size已定义
predict_model.to(device)
# 加载已保存的模型参数
predict_model.load_state_dict(torch.load('model_trunks12_parameters.pth', map_location='cuda:0'))
models = []
import math
for n in range(0, output_size ):
    model = imagebind_model.imagebind_huge(pretrained=True, audio_num_blocks=n)
    device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 计算GPU索引
    gpu_index = n // (output_size // 3)
    model.to(device_ids[gpu_index])
    
    models.append(model)
layers=[]
for embedding_item in embeddings[ModalityType.AUDIO]:
        embedding_item=embedding_item.to(device)
        layer=predict_model(embedding_item.float())
        _, layer1 = torch.max(layer, 0)
        layers.append(layer1)
import pickle

# 定义文件路径
file_path = "layers.pkl"

# 保存数据到文件
with open(file_path, 'wb') as f:
    pickle.dump(layers, f)
embedding_dynamic={}

with torch.no_grad():
  for i in range(len(embeddings[ModalityType.AUDIO])):
        
        gpu_index = layers[i] // (output_size // 3)
        inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data2(audio_path[i],device=device_ids[gpu_index])
        }
        
        current_embeddings = models[layers[i]](inputs)[ModalityType.AUDIO]

        if embedding_dynamic:
                embedding_dynamic[ModalityType.AUDIO] = torch.cat([embedding_dynamic[ModalityType.AUDIO], current_embeddings.to(embedding_dynamic[ModalityType.AUDIO].device)], dim=0)
        else:
                embedding_dynamic[ModalityType.AUDIO] = current_embeddings

        del current_embeddings
torch.save({
        'audio_embeddings': embedding_dynamic[ModalityType.AUDIO]
    }, f'parameters/dynamic/image/embeddings_{N}.pth')

