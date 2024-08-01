import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyCNN(nn.Module):
    def __init__(self, n_outputs):
        super(PolicyCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(32 * 6 * 7, n_outputs)  # Assuming input size is 8x8

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten except batch dim
        x = self.fc1(x)
        x = torch.softmax(x, dim=1)
        return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return F.relu(out)
    
# class PolicyCNN(nn.Module):
#     def __init__(self, n_outputs):
#         super(PolicyCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # (N, 1, 6, 7) -> (N, 64, 6, 7)
#         self.bn1 = nn.BatchNorm2d(64)
        
#         # Increase the number of residual blocks
#         self.res_blocks = nn.Sequential(
#             *[ResidualBlock(64) for _ in range(4)]
#         )
        
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 64, 6, 7) -> (N, 64, 1, 1)
#         self.fc1 = nn.Linear(64, n_outputs)

#     def forward(self, x):  # x: (N, 1, 6, 7)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.res_blocks(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x) # (N, n_outputs)
#         x = torch.softmax(x, dim=1)
#         return x