import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        # Input size: (3, 10, 10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Output size: (8, 10, 10)
        self.bn1 = nn.BatchNorm2d(8)
        # Output size: (8, 10, 10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size: (8, 5, 5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Output size: (16, 5, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # Output size: (16, 5, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size: (16, 2, 2) <-- Corrected
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output size: (32, 2, 2)
        self.bn3 = nn.BatchNorm2d(32)
        # Output size: (32, 2, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size: (32, 1, 1)
        self.fc1 = nn.Linear(32 * 1 * 1, 128)  # Adjust the input size here
        # Output size: (128)
        self.fc2 = nn.Linear(128, 64)
        # Output size: (64)
        self.fc3 = nn.Linear(64, 32)
        # Output size: (32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # Convolution and BatchNorm
        x = self.pool1(x)  # MaxPooling
        x = F.leaky_relu(self.bn2(self.conv2(x)))  # Convolution and BatchNorm
        x = self.pool2(x)  # MaxPooling
        x = F.leaky_relu(self.bn3(self.conv3(x)))  # Convolution and BatchNorm
        x = self.pool3(x)  # MaxPooling
        x = x.view(-1, 32 * 1 * 1)  # Flatten the feature map
        x = F.leaky_relu(self.fc1(x))  # Fully Connected with Activation
        x = F.leaky_relu(self.fc2(x))  # Fully Connected with Activation
        x = F.leaky_relu(self.fc3(x))  # Fully Connected with Activation
        x = self.fc4(x)  # Output layer
        return x


    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


model = QNet()