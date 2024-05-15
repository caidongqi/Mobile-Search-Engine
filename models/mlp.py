import torch.nn.functional as F
import torch.nn as nn
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


class MyModel3(nn.Module):      
    def __init__(self, n,num_classes):      
        super(MyModel3, self).__init__()      
        
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