import torch
import random

def random_flip(volume):
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[2])  
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[3])  
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[1])  
    return volume

def random_rotate(volume):
    k = random.randint(0, 3)
    volume = torch.rot90(volume, k, dims=[2,3])  
    return volume

def add_noise(volume, sigma=0.01):
    noise = torch.randn_like(volume) * sigma
    return volume + noise

def transform(volume):
    volume = random_flip(volume)
    volume = random_rotate(volume)
    volume = add_noise(volume)
    return volume
