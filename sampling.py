import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch as torch
import torchvision as tv
from torchsummary import summary

from MNIST import MNIST_3s
from diffusion import *
from process_data import *
from unet import *

img_size = 32

# Hyperparameters
batch_size, T, epochs, learning_rate = hyperparams()

# Load model
model = UNet()
model.load_state_dict(torch.load("weights/UNet_diff.pth", weights_only = True))
diff_model = diffusion(timesteps = T)

xt = torch.randn(1, 1, img_size, img_size)

fig, ax = plt.subplots(1, 11)
for t in range(0, T)[::-1]:
    t_tensor = torch.full((1,), t)
    xt = diff_model.backward_step(xt, t_tensor, model)
    xtimg = tensor_to_pil(xt)
    if (t+1)%100 == 0 or t == 0:
        plt.imshow(xtimg, cmap = "gray")
        plt.axis(False)
        plt.savefig(f"results/t_{t + 1}.png")
    
