import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(6, 12, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2) 

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(12, 12)
        
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(12, num_classes)
    def forward(self, x):
        
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x