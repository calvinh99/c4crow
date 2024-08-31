import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvDQN(nn.Module):
    def __init__(self, n_channels=2, n_rows=6, n_cols=7):
        # the input is Batch size, num feature dims, num rows, num cols
        # first channel will always be RL player, and second channel will always be opponent player
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1) # 2x6x7 -> 32x6x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x6x7 -> 64x6x7
        self.fc1 = nn.Linear(64 * n_rows * n_cols, 512) # 64x6x7 -> 512
        self.fc2 = nn.Linear(512, n_cols) # 512 -> 7

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
