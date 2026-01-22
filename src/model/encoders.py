import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        c2 = self.conv1(x)
        p2 = self.pool(c2)
        c3 = self.conv2(p2)
        p3 = self.pool(c3)
        c4 = self.conv3(p3)
        p4 = self.pool(c4)
        c5 = self.conv4(p4)
        return [c2, c3, c4, c5]
