import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import argparse

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(process)d - %(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

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

def evaluate_nn(model, source_test, X_test, y_test, layer_num, device, S, save=False):
    # 初始化测试数据和结果的字典
    X_test_dict = {}
    y_test_dict = {}
    accuracy_dict = {}
    predicted_layer_count = {layer_key: [] for layer_key in range(1, layer_num + 1)}

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
        for layer_key in range(1, layer_num + 1):
            if layer_key in X_test_dict:
                outputs = model(torch.tensor(X_test_dict[layer_key]).to(device).float())
                _, predicted = torch.max(outputs, 1)
                accuracy_dict[layer_key] = (predicted == torch.tensor(y_test_dict[layer_key]).to(device)).sum().item() / len(y_test_dict[layer_key])
                logging.info(f'Accuracy for layer {layer_key}: {accuracy_dict[layer_key]:.2f}')
                
                # 记录每个样本的预测层数
                for pred in predicted:
                    predicted_layer_count[layer_key].append(pred.item())

        if save:
            # 将结果写入CSV文件
            with open(f'model_coco_lora.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['error_bar'] + [f'layer{layer_key}' for layer_key in sorted(accuracy_dict.keys())]
                writer.writerow(header)
                row = [total_accuracy] + [accuracy_dict[layer_key] for layer_key in sorted(accuracy_dict.keys())]
                writer.writerow(row)
        
        # 保存模型参数
        torch.save(model.state_dict(), f'parameters/image/coco/model/image_S={S}_train_10_lora_bs128.pth')

    # 计算每一层的平均预测层数
    avg_predicted_layer = {layer_key: np.mean(predicted_layer_count[layer_key]) for layer_key in predicted_layer_count}
    for layer_key in avg_predicted_layer:
        logging.info(f'Average predicted layer for layer {layer_key}: {avg_predicted_layer[layer_key]:.2f}')

    return accuracy_dict, total_accuracy, avg_predicted_layer


device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--S", default=90,type=int, help="Number of S")


# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
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
root=f"parameters/image/coco/train_10" # f"parameters/image/coco"
embeddings_dict = {}

for i in range(1, layer_num + 1):
    embeddings_dict[str(i)] = torch.load(os.path.join(root, f'embeddings_{i}_lora.pth'),map_location=torch.device(device))['vision_embeddings']
    
layers = np.loadtxt(f'./results/coco_lora_train_10/R{S}/layers.txt')   #  f'./results/coco_lora/R{S}/layers.txt'
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
num_epochs = 100
batch_size = 128
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        # 前向传播
        outputs = model(inputs)
        # 将目标值移动到与模型输出相同的设备上
        targets = targets.to(device)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            logging.info(f'Batch [{i }/{len(X_train_tensor)}] Loss: {loss.item():.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    evaluate_nn(model, source_test, X_test, y_test, layer_num, device, S, save=False)

    