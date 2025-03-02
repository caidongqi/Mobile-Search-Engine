import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
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

# 初始化模型
device = "cuda:3" if torch.cuda.is_available() else "cpu"
input_size = 175872  # 根据embedding的大小确定输入层大小
output_size = 12  # 根据层数的范围确定输出层大小
model = MyModel(input_size, output_size)
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 加载数据
embeddings_1 = torch.load('embeddings_1_trunks.pth')['audio_embeddings']
embeddings_2 = torch.load('embeddings_2_trunks.pth')['audio_embeddings']
embeddings = torch.cat((embeddings_1, embeddings_2), dim=0)
file = '/data/air/pc/Mobile-Search-Engine/results/clotho/R10/layers_min.txt'
layers = np.loadtxt(file)
layers = np.concatenate((layers, layers), axis=0)
# 划分训练集和测试集


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, layers, test_size=0.2, random_state=42)
X_train=X_train.to(device)
X_test=X_test.to(device)
# X_test_1=X_test_1.to(device)
# X_test_2=X_test_2.to(device)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()

# 训练模型
num_epochs = 10
batch_size = 32
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
with torch.no_grad():
    outputs = model(torch.tensor(X_test).float())
    _, predicted = torch.max(outputs, 1)
    accuracy1 = (predicted == torch.tensor(y_test).to(device)).sum().item() / len(y_test)
    print(f'Accuracy_error_bar: {accuracy1:.2f}')
    with open('output1.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer1'])
        
        writer.writerow([accuracy1])
    

    
    torch.save(model.state_dict(), 'model_trunks1&2_parameters.pth')
   