import os
import numpy as np
from PIL import Image

import torch as torch
import torchvision as tv

def expand_image(image):
    """
    For later purposes, it's more convinient for the images to have a 32x32 resolution instead of 28x28,
    as 32 is a power of 2
    """
    array = np.array(image)
    array = np.pad(array, 2)
    return Image.fromarray(array)

class MNIST_3s(torch.utils.data.Dataset):
    """
    Samples of the MNIST only containing the 3s.
    """

    def __init__(self, transformation, root = "mnist_png/training/1/"):
        self.root = root
        self.transformation = transformation
        self.list = os.listdir(self.root)
    
    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        image = Image.open(self.root + self.list[idx])
        image = expand_image(image)
        image = self.transformation(image)
        return image
    
    def shape(self):
        image = Image.open(self.root + self.list[0])
        image = expand_image(image)
        image = self.transformation(image)
        return image.shape
    
    
