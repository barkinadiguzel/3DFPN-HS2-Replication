import matplotlib.pyplot as plt
import numpy as np

def overlay_slice(volume, mask, slice_idx):
    vol_slice = volume[slice_idx].numpy()
    mask_slice = mask[slice_idx].numpy()

    plt.imshow(vol_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='Reds', alpha=0.3)
    plt.show()
