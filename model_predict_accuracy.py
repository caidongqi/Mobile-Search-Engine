import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import argparse
import logging
import matplotlib.pyplot as plt

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


# 解析命令行参数
args = parser.parse_args()
# 获取 audio_num_blocks 的值
S=args.S
version='model_acc'
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
model_parameter=f'parameters/image/coco/model/image_S={S}_val_min_v1.pth'
model = MyModel(input_size, output_size)
model.to(device)
model.load_state_dict(torch.load(model_parameter,map_location=device))
# 定义损失函数和优化器

import os
layer_num = 32
root=f"parameters/image/coco/val"
embeddings_dict = {}

for i in range(1, layer_num + 1):
    embeddings_dict[str(i)] = torch.load(os.path.join(root, f'embeddings_{i}_trunk_lora.pth'),map_location=torch.device(device))['vision_embeddings']
    
layers = np.loadtxt(f'./results/coco_lora_val/R{S}/layers_min.txt')  
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


# 初始化测试数据和结果的字典
X_test_dict = {}
y_test_dict = {}
accuracy_dict = {}
layer_dict={}
X_test=embeddings
y_test=layers
# 用torch.no_grad()来停用梯度计算，减少内存消耗和加速计算
with torch.no_grad():
    # 动态创建测试集
    for i, item in enumerate(source_indicator):
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
    np.savetxt('predict.txt',predicted.cpu())
    np.savetxt('target.txt',torch.tensor(y_test).cpu())
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
    with open(f'model_coco_lora_val_N_S_min.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['S']+ [f'layer{layer_key}' for layer_key in sorted(accuracy_dict.keys())]+['error_bar'] 
        writer.writerow(header)
        row = [S]+ [accuracy_dict[layer_key] for layer_key in sorted(accuracy_dict.keys())]+[total_accuracy] 
        writer.writerow(row)
        row = [S] + [layer_dict[layer_key] for layer_key in sorted(layer_dict.keys())]+[total_accuracy]
        writer.writerow(row)

        # 创建一个新的图
    # 假设 layer_num 是层的总数
    layer_keys = list(range(1, layer_num + 1))

    # 准备数据
    accuracies = [accuracy_dict[key] for key in layer_keys]
    layer_means = [layer_dict[key] for key in layer_keys]
    fig, ax1 = plt.subplots()

    # 绘制准确率曲线
    ax1.set_xlabel('Layer Key')
    ax1.set_ylabel('Accuracy', color='tab:red')
    ax1.plot(layer_keys, accuracies, 'r-', label='Accuracy')  # 'r-' 表示红色实线
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 创建一个共享x轴的第二个轴用于绘制层均值
    ax2 = ax1.twinx()  # 共享x轴
    ax2.set_ylabel('Mean Layer', color='tab:blue')
    ax2.plot(layer_keys, layer_means, 'b-', label='Mean Layer')  # 'b-' 表示蓝色实线
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 设置图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 显示图表
    plt.title('Accuracy and Mean Layer per Layer Key')
    plt.show()
    # 保存模型参数
    #model: N=32 lora S=10
    # torch.save(model.state_dict(), f'parameters/image/coco/model/image_S={S}_coco_val.pth')
   