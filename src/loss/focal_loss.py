import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-6, 1-1e-6)
        bce_loss = - (targets * torch.log(inputs) + (1-targets) * torch.log(1-inputs))
        pt = torch.where(targets==1, inputs, 1-inputs)
        focal_factor = (1 - pt) ** self.gamma
        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
