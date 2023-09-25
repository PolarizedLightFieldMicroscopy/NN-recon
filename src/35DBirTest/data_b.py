"""Create dataset classes to import the data into the appropriate format."""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread


class BirDataB(Dataset):
    """A PyTorch dataset to load polarized light field images and birefringent objects
    
    """
    def __init__(
        self,
        root_dir,
        source_complex_norm=False,
        target_idx=None,
        target_channel=None,
        split="train",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.samples = os.listdir(root_dir)
        self.source = os.listdir(os.path.join(self.root_dir, "images"))
        self.target = os.listdir(os.path.join(self.root_dir, "objects"))
        self.source_complex_norm = source_complex_norm
        self.target_idx = target_idx
        self.target_channel = target_channel

        num_train = int(0.70 * len(self.source))
        num_val = int(0.15 * len(self.source))

        if split == "train":
            self.source = self.source[:num_train]
            self.target = self.target[:num_train]
        elif split == "val":
            self.source = self.source[num_train : num_train + num_val]
            self.target = self.target[num_train : num_train + num_val]
        elif split == "test":
            self.source = self.source[num_train + num_val :]
            self.target = self.target[num_train + num_val :]
            
    # get the total number of samples
    def __len__(self):
        return len(self.source)

    # get the i'th sample
    def __getitem__(self, index):
        source = imread(os.path.join(self.root_dir, "images", self.source[index]))
        target = imread(os.path.join(self.root_dir, "objects", self.target[index]))

        if self.source_complex_norm:
            source = self.source_transform(source)

        if self.target_idx is not None:
            target = target[self.target_idx]
        
        return source, target

    def source_transform(self, img):
        return torch.from_numpy(img).float()
