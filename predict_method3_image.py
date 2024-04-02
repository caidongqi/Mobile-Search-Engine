import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
from sklearn.model_selection import train_test_split
 
  
class ConvNet(nn.Module):      
    def __init__(self, n,num_classes):      
        super(ConvNet, self).__init__()      
        
        input_length = 1024  
        for i in range(3):  # For each conv and pool layer  
            input_length = (input_length - (3 - 1)) // 2  
        fc1_input_dim = 128 * input_length  # 128 is the number of output channels of the last conv layer  
        self.conv1 = nn.Conv1d(in_channels=n, out_channels=32, kernel_size=3)      
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)      
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)      
          
        self.pool = nn.MaxPool1d(kernel_size=2)      
          
        self.fc1 = nn.Linear(fc1_input_dim, 128) # 1024 - 2*3 = 1018, 1018//2 = 509, 509 - 2*3 = 503, 503//2 = 251, 251 - 2*3 = 245, 245//2 = 122, 128 * 122 = 15616  
        self.fc2 = nn.Linear(128, num_classes)  
          
    def forward(self, x):      
        x = F.relu(self.conv1(x))      
        x = self.pool(x)      
        x = F.relu(self.conv2(x))      
        x = self.pool(x)      
        x = F.relu(self.conv3(x))      
        x = self.pool(x)      
          
        x = x.view(x.size(0), -1)      
        x = F.relu(self.fc1(x))      
        x = self.fc2(x)      
        return F.log_softmax(x, dim=1)     

import os
# 加载数据
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--device", type=str, default="cuda:5", help="Device to use (cuda:2 or cpu)")
parser.add_argument("--layer_num", default=1,type=int, help="Number of audio blocks")
# 解析命令行参数
args = parser.parse_args()

# 获取 audio_num_blocks 的值
layer_num=args.layer_num
device = args.device

#layer_num = 1
root = "/data/air/pc/Mobile-Search-Engine/parameters/image/trunks+post"
embeddings_dict = {}

for i in range(1, layer_num + 1):
    embeddings_dict[str(i)] = torch.load(os.path.join(root, f'embeddings_{i}.pth'))['audio_embeddings']

embeddings_list = []

# 假设 embeddings_dict 是你的字典
for i in range(1, layer_num + 1):
    embeddings_list.append(embeddings_dict[str(i)].cpu().numpy())

concatenated_embeddings = np.stack(embeddings_list, axis=1)

file = '/data/air/pc/Mobile-Search-Engine/results/imagenet/5000/R10/layers_min.txt'
layers = np.loadtxt(file)
X_train, X_test, y_train, y_test = train_test_split(concatenated_embeddings, layers, test_size=0.2, random_state=42)

# Initialize the model
device = "cuda:3" if torch.cuda.is_available() else "cpu"
input_size = layer_num  # Assuming input size
outputs_label=32
model = ConvNet(input_size, outputs_label)  # Input shape modified to match the input size
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume embeddings_dict is already defined
# Load and preprocess data

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(concatenated_embeddings, layers, test_size=0.2, random_state=42)

# Convert data to tensors and move to device
X_train_tensor = torch.tensor(X_train).float().to(device)
y_train_tensor = torch.tensor(y_train).long().to(device)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Calculate accuracy
with torch.no_grad():
    outputs = model(torch.tensor(X_test).float().to(device))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.tensor(y_test).to(device)).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.2f}')
    
    with open('output3_image.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header =  [f'layer{layer_num}' ]
        writer.writerow(header)
        row = [accuracy]
        writer.writerow(row)
    # Save model parameters
   
