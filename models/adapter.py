# # # # adapter.py  
from torch import nn  
class Adapter(nn.Module):  
    def __init__(self):  
        super(Adapter, self).__init__()  
        self.fc = nn.Linear(1024, 512*7*7)  
        self.up_sample = nn.Sequential(  
            nn.ReLU(),  
            nn.BatchNorm2d(512),  
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),  
            nn.BatchNorm2d(256),  
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),  
            nn.BatchNorm2d(128),  
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),  
            nn.BatchNorm2d(64),  
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 新增的转置卷积层  
            nn.ReLU(),  
            nn.BatchNorm2d(32),  
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  
            nn.Tanh()  
        )  
  
    def forward(self, x):  
        x = self.fc(x)  
        x = x.view(x.size(0), 512, 7, 7)  # reshape to 4D tensor  
        x = self.up_sample(x)  
        return x  