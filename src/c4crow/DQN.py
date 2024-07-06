import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
N is batch_size -> 96
C is number of channels -> 1
H is number of rows -> 6
W is number of cols -> 7
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # (N, C, H, W) -> (N, C, H, W)
        self.bn1 = nn.BatchNorm2d(in_channels)  # (N, C, H, W) -> (N, C, H, W)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # (N, C, H, W) -> (N, C, H, W)
        self.bn2 = nn.BatchNorm2d(in_channels)  # (N, C, H, W) -> (N, C, H, W)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))  # (N, C, H, W) -> (N, C, H, W)
        out = self.bn2(self.conv2(out))  # (N, C, H, W) -> (N, C, H, W)
        out += residual  # (N, C, H, W) -> (N, C, H, W), this is the skip connection
        return F.relu(out)  # (N, C, H, W) -> (N, C, H, W)

class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (N, 1, 6, 7) -> (N, 32, 6, 7)
        self.bn1 = nn.BatchNorm2d(32)  # (N, 32, 6, 7) -> (N, 32, 6, 7)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),  # (N, 32, 6, 7) -> (N, 32, 6, 7)
            ResidualBlock(32),  # (N, 32, 6, 7) -> (N, 32, 6, 7)
            ResidualBlock(32)   # (N, 32, 6, 7) -> (N, 32, 6, 7)
        )
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (N, 32, 6, 7) -> (N, 64, 6, 7)
        self.bn2 = nn.BatchNorm2d(64)  # (N, 64, 6, 7) -> (N, 64, 6, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 64, 6, 7) -> (N, 64, 1, 1)
        
        self.fc1 = nn.Linear(64, 64)  # (N, 64) -> (N, 64)
        self.fc2 = nn.Linear(64, outputs)  # (N, 64) -> (N, outputs)

    def forward(self, x):  # x: (N, 1, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))  # (N, 1, 6, 7) -> (N, 32, 6, 7)
        x = self.res_blocks(x)  # (N, 32, 6, 7) -> (N, 32, 6, 7)
        x = F.relu(self.bn2(self.conv2(x)))  # (N, 32, 6, 7) -> (N, 64, 6, 7)
        x = self.pool(x)  # (N, 64, 6, 7) -> (N, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 64, 1, 1) -> (N, 64)
        x = F.relu(self.fc1(x))  # (N, 64) -> (N, 64)
        return self.fc2(x)  # (N, 64) -> (N, outputs)

class DQN2(nn.Module):
    def __init__(self, outputs):
        super(DQN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # (N, 1, 6, 7) -> (N, 64, 6, 7)
        self.bn1 = nn.BatchNorm2d(64)  # (N, 64, 6, 7) -> (N, 64, 6, 7)
        
        # Increase the number of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(10)]  # 10 residual blocks
        )
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (N, 64, 6, 7) -> (N, 128, 6, 7)
        self.bn2 = nn.BatchNorm2d(128)  # (N, 128, 6, 7) -> (N, 128, 6, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 128, 6, 7) -> (N, 128, 1, 1)
        
        self.fc1 = nn.Linear(128, 128)  # (N, 128) -> (N, 128)
        self.fc2 = nn.Linear(128, outputs)  # (N, 128) -> (N, outputs)

    def forward(self, x):  # x: (N, 1, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))  # (N, 1, 6, 7) -> (N, 64, 6, 7)
        x = self.res_blocks(x)  # (N, 64, 6, 7) -> (N, 64, 6, 7)
        x = F.relu(self.bn2(self.conv2(x)))  # (N, 64, 6, 7) -> (N, 128, 6, 7)
        x = self.pool(x)  # (N, 128, 6, 7) -> (N, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 128, 1, 1) -> (N, 128)
        x = F.relu(self.fc1(x))  # (N, 128) -> (N, 128)
        return self.fc2(x)  # (N, 128) -> (N, outputs)