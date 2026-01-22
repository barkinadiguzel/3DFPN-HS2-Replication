import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import Encoder3D
from .feature_pyramid import FeaturePyramid3D
from .hs2_net import HS2Net
from .lhi_utils import compute_lhi

class FPN_HS2_Model(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = Encoder3D(in_channels)
        self.fpn = FeaturePyramid3D()
        self.hs2 = HS2Net()

    def forward(self, x):
        features = self.encoder(x)
        pyramid = self.fpn(features)
        candidates = []
        for p in pyramid:  
            B,C,D,H,W = p.shape
          
            mask = (p > 0.5)
            indices = mask.nonzero(as_tuple=False)
            for idx in indices:
                candidates.append({
                    'pyramid_level': p,
                    'coord': idx[:3],  
                    'feature': p[idx[0], :, idx[2], idx[3], idx[4]]
                })

        outputs = []
        for c in candidates:
            z,y,x = c['coord']
            cube = c['pyramid_level'][:, :, max(0,z-2):z+2, max(0,y-2):y+2, max(0,x-2):x+2]
            lhi = compute_lhi(cube.squeeze(0))  
            out = self.hs2(lhi)
            outputs.append(out)

        if len(outputs) == 0:
            return None
        return torch.stack(outputs)
