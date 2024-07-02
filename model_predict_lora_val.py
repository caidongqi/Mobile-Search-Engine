import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import argparse

import logging


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # 输入层到隐藏层
        self.fc2 = nn.Linear(256, 64)  # 隐藏层到输出层
        self.fc3 = nn.Linear(64, output_size)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--S", default=10,type=int, help="Number of S")
parser.add_argument("--version", default='train_model_flicker',type=str, help="The tags you want to describe the experiment")
parser.add_argument("--root", default='parameters/image/flickr8k/val',type=str, help="The embeddings path")
parser.add_argument("--embeddings_file", default='embeddings_{i}.pth',type=str, help="The embeddings file")
parser.add_argument("--layers_file", default='./results/flickr8k_lora_val_nohead/R{S}/layers.txt',type=str, help="The layers file")
parser.add_argument("--outputs_path", default='parameters/image/flickr8k/val/model/image_S={S}_val_v2.pth',type=str, help="The output path")
parser.add_argument("--csv_path", default='model_flicker_val.csv',type=str, help="The output path")


# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
version=args.version
root=args.root
embeddings_file=args.embeddings_file
layers_file=args.layers_file
csv_path=args.csv_path
outputs_path=args.outputs_path
import time
# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间
local_time = time.localtime(timestamp)
# 格式化本地时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(f'logs/model/S={S}_{version}_{formatted_time}.log')], 
                    force=True)

input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 32  # 根据层数的范围确定输出层大小
model = MyModel(input_size, output_size)
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import os
layer_num = 32
# root = "parameters/image/trunks+post"

embeddings_dict = {}

for i in range(1, layer_num + 1):
    embeddings_dict[str(i)] = torch.load(os.path.join(root, embeddings_file.format(i=i)),map_location=torch.device(device))['vision_embeddings']

layers_file=layers_file.format(S=S)
layers = np.loadtxt(layers_file)  
# logging.info(layers)
layers = np.concatenate(tuple((layers) for i in range(layer_num)), axis=0)
# 根据layers值获取对应的embeddings
all_embeddings = []
source_indicator = []
for layer_value in range(1, layer_num+1):
    if str(int(layer_value)) in embeddings_dict:
        embeddings = embeddings_dict[str(int(layer_value))]
        all_embeddings.append(embeddings)
        source_indicator += [int(layer_value)] * len(embeddings)

# 将所有的embeddings拼接成一个大的tensor
embeddings = torch.cat([tmp.to(device) for tmp in all_embeddings], dim=0)
source_indicator = np.array(source_indicator)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(embeddings,layers, source_indicator, test_size=0.2, random_state=42)
print(y_train.min(), y_train.max())

X_train=X_train.to(device)
X_test=X_test.to(device)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()

# 训练模型
num_epochs = 50
batch_size = 4
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]
        targets = targets.to(device)
        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == targets.to(device)).sum().item() / len(targets)
        
        # 计算损失
        loss = criterion(outputs, targets)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            logging.info(f'Batch [{i / batch_size}/{len(X_train_tensor)/batch_size}] Loss: {loss.item():.4f}')
            logging.info(f'Batch [{i / batch_size}/{len(X_train_tensor)/batch_size}] Acc: {accuracy:.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 初始化测试数据和结果的字典
X_test_dict = {}
y_test_dict = {}
accuracy_dict = {}
layer_dict={}
# 用torch.no_grad()来停用梯度计算，减少内存消耗和加速计算
with torch.no_grad():
    # 动态创建测试集
    for i, item in enumerate(source_test):
        layer_key = int(item)  # 将item转换为整数键
        if layer_key not in X_test_dict:
            X_test_dict[layer_key] = []
            y_test_dict[layer_key] = np.array([])
        
        # 将X_test转换为numpy.ndarray对象并添加到对应的列表
        X_test_np = X_test[i].cpu().numpy()
        X_test_dict[layer_key].append(X_test_np)
        y_test_dict[layer_key] = np.append(y_test_dict[layer_key], y_test[i])
    
    # 总体精度计算
    outputs = model(torch.tensor(X_test).to(device).float())
    _, predicted = torch.max(outputs, 1)
    total_accuracy = (predicted == torch.tensor(y_test).to(device)).sum().item() / len(y_test)
    logging.info(f'Total Accuracy: {total_accuracy:.2f}')
   
    # 分层精度计算
    for layer_key in range(1, layer_num+1):
        outputs = model(torch.tensor(X_test_dict[layer_key]).to(device).float())
        _, predicted = torch.max(outputs, 1)
        accuracy_dict[layer_key] = (predicted == torch.tensor(y_test_dict[layer_key]).to(device)).sum().item() / len(y_test_dict[layer_key])
        layer_dict[layer_key]=torch.mean(predicted.float())
        logging.info(f'Accuracy for layer {layer_key}: {accuracy_dict[layer_key]:.2f}')
        logging.info(f'mean layer {layer_key}: {layer_dict[layer_key]:.2f}')

    # 将结果写入CSV文件
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['S']+['error_bar'] + [f'layer{layer_key}' for layer_key in sorted(accuracy_dict.keys())]
        writer.writerow(header)
        row = [S]+[total_accuracy] + [accuracy_dict[layer_key] for layer_key in sorted(accuracy_dict.keys())]
        writer.writerow(row)
        row = [S]+[total_accuracy] + [layer_dict[layer_key] for layer_key in sorted(layer_dict.keys())]
        writer.writerow(row)

   
    # 保存模型参数
    #model: N=32 lora S=10
    outputs_path=outputs_path.format(S=S)
    torch.save(model.state_dict(), outputs_path)
   