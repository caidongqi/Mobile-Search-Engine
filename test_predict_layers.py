import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

input_size = 1024  # 根据embedding的大小确定输入层大小
output_size = 12  # 根据层数的范围确定输出层大小
new_model = MyModel(input_size, output_size)

# 加载之前保存的模型参数
new_model.load_state_dict(torch.load('model_trunks1&2_parameters.pth'))

device = "cuda:3" if torch.cuda.is_available() else "cpu"
new_model.to(device)
T_logits=[]
F_logits=[]
embeddings = torch.load('parameters/audio/trunks+post/embeddings_1.pth')['audio_embeddings']
file = '/data/air/pc/Mobile-Search-Engine/results/clotho/R10/layers_min.txt'
layers = np.loadtxt(file)
embeddings.to(device)

outputs = new_model(torch.tensor(embeddings).to(device).float())
probabilities = F.softmax(outputs, dim=1)

# 找到最大概率对应的类别及其置信度
confidence, predicted_class = torch.max(probabilities, 1)
_, predicted = torch.max(outputs, 1)
accuracy = (predicted == torch.tensor(layers).to(device)).sum().item() / len(layers)
for i in range(len(layers)):
    if(layers[i]==predicted[i].item()):
        T_logits.append(confidence[i].item())
    else:
        F_logits.append(confidence[i].item())
print(sum(T_logits)/len(T_logits))
print(sum(F_logits)/len(F_logits))
print(accuracy)