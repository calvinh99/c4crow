import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvDQN(nn.Module):
    def __init__(self, n_channels=2, n_rows=6, n_cols=7):
        # the input is Batch size, num feature dims, num rows, num cols
        # first channel will always be RL player, and second channel will always be opponent player
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1) # 2x6x7 -> 32x6x7
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x6x7 -> 64x6x7
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 64x6x7 -> 128x6x7
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 128x6x7 -> 256x6x7
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * n_rows * n_cols, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_cols)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
