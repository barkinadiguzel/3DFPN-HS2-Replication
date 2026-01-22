import torch

def compute_lhi(volume_slices, threshold=30, tau=5):
    S, H, W = volume_slices.shape
    lhi = torch.zeros((H, W), dtype=torch.float32)

    for s in range(1, S):
        diff = torch.abs(volume_slices[s] - volume_slices[s-1])
        psi = (diff > threshold).float()
        lhi = torch.where(psi==1, torch.tensor(tau), torch.clamp(lhi-1, min=0))
    return lhi.unsqueeze(0).unsqueeze(0)  
