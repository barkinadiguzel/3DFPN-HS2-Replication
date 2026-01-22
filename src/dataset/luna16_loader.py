import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
import numpy as np

class LUNA16Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        volume = nib.load(path).get_fdata()  
        volume = torch.from_numpy(volume).float().unsqueeze(0)  

        if self.transform:
            volume = self.transform(volume)

        # Dummy mask for demo (in practice, load annotated mask)
        mask = torch.zeros_like(volume)
        return volume, mask
