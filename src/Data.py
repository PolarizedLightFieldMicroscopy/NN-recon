"""Create dataset classes to import the data into the appropriate format."""

from torch.utils.data import Dataset, DataLoader

# import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import tifffile
from tifffile import imread


class RestoratorsDataset(Dataset, ABC):
    """An abstract PyTorch dataset class to serve as a base class for
    our custom transformer datasets."""

    def __init__(self, root_dir):
        assert self.root_dir is not None
        return None

    @abstractmethod
    def __len__(self):
        return None

    @abstractmethod
    def __getitem__(self, index):
        """Describes what happens when you index into a Dataset
        Parameters:
            index: int
        Returns: a 2-tuple of 3D tensors of the form (C, H, W)
        """
        return None


class BirefringenceDataset(RestoratorsDataset):
    """A PyTorch dataset to load polarized light field images and birefringent objects"""

    def __init__(
        self,
        root_dir,
        source_norm=False,
        target_norm=False,
        transform=None,
        split="train",
    ):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the parent directory of samples
        self.source = os.listdir(os.path.join(self.root_dir, "images"))  # source domain
        self.target = os.listdir(
            os.path.join(self.root_dir, "objects")
        )  # target domain
        self.img_transform = (
            self.img_transform
        )  # transformations to apply to raw LF image only
        self.source_norm = source_norm
        self.target_norm = target_norm
        self.transform = transform  # transformations for augmentations
        #  transformations to apply just to inputs
        # self.input_transform # transforms.ToTensor() only works on smaller dim images
        num_train = int(
            0.60 * len(self.source)
        )  # reducing this will speed up the epochs when training
        num_test = int(0.16 * len(self.source))
        num_val = int(0.24 * len(self.source))
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

    # fetch the training sample given its index
    def __getitem__(self, idx):
        source_path = os.path.join(self.root_dir, "images", self.source[idx])
        # We'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        source = tifffile.imread(source_path)
        if self.source_norm:
            source = self.source_transform(source)
        source = self.numpy2tensor(source).to(torch.float32)
        target_path = os.path.join(self.root_dir, "objects", self.target[idx])
        target = tifffile.imread(target_path)
        if self.target_norm:
            target = self.target_transform(target)
        target = self.numpy2tensor(target).to(torch.float32)
        return source, target

    def img_transform(self, image):
        """Transforms, normally in a simple way, the source data."""
        return image

    def numpy2tensor(self, array):
        return torch.from_numpy(array)

    def source_transform(self, source):
        """Normalize the retardance and azimuth values"""
        # pinholes = source.shape[0] / 2
        # delta = source[:pinholes, ...]
        # azim = source[pinholes:, ...]
        # new_source = np.zeros(source.shape)
        new_source = np.sin(source)
        return new_source

    def target_transform(self, target):
        """Normalize the birefringence values"""
        new_target = np.zeros(target.shape)
        delta_n = target[0, ...]
        new_target[0, ...] = (delta_n - 0.005) / 0.01 + 0.5
        # optic axis elements are still between -1 and 1, not 0 and 1
        return new_target


def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset), 1)[0]
    img, mask = dataset[idx]
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def load():
    TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
    train_data = BirefringenceDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

    VAL_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
    val_data = BirefringenceDataset(VAL_DATA_PATH)
    val_loader = DataLoader(val_data, batch_size=5)
    return train_loader, val_loader


if __name__ == "__main__":
    # train_loader, val_loader = load()
    TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"

    train_data = BirefringenceDataset(
        TRAIN_DATA_PATH, split="test", source_norm=True, target_norm=True
    )
    train_data[0]  # pair 0
    train_data[1]  # pair 1
