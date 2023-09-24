'''Create dataset classes to import the data into the appropriate format.'''
from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

class BirData(Dataset):
    def __init__(self, root_dir, source_norm=False, target_norm=False, transform=None, split='train'):
        super().__init__()
        
