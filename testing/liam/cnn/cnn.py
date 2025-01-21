# model.py

import torch
import torch.nn as nn

class Music1DCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super(Music1DCNN, self).__init__()
        
        # 1st 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 2nd 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * (input_length // 4), 100)  # Adjust input_length accordingly
        self.fc2 = nn.Linear(100, num_classes)  # Output layer for classification
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * (x.size(2)))  # Flatten the feature maps
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
