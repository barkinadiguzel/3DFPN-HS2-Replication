import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramid3D(nn.Module):
    def __init__(self, channels=[32,64,128,256]):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(c, 128, 1) for c in channels
        ])
        self.top_down_convs = nn.ModuleList([
            nn.Conv3d(128, 128, 3, padding=1) for _ in channels
        ])

    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode='trilinear', align_corners=False)
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode='trilinear', align_corners=False)
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode='trilinear', align_corners=False)

        p5 = self.top_down_convs[3](p5)
        p4 = self.top_down_convs[2](p4)
        p3 = self.top_down_convs[1](p3)
        p2 = self.top_down_convs[0](p2)
        return [p2, p3, p4, p5]
