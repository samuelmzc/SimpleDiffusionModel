import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch as torch
import torchvision as tv
from torchsummary import summary

from MNIST import MNIST_3s
from diffusion import *
from process_data import *
from unet import *

img_size = 32

# Hyperparameters
batch_size, T, epochs, learning_rate, _ = hyperparams()

# Load model
model = UNet()
model.load_state_dict(torch.load("weights/UNet_diff.pth", weights_only = True))
diff_model = diffusion_cosine(timesteps = T)

# print information
verbose = False

if verbose == True:
    print(f"=======================================================")
    print(f"Hyperparameters")
    print(f"-------------------------------------------------------")
    print(f"Batch size          ==> {batch_size}")
    print(f"Diffusion timesteps ==> {T}")
    print(f"Embedding dimension ==> {embedding_dim}")
    print(f"Epochs              ==> {epochs}")
    print(f"Learning rate       ==> {learning_rate}")
    print(f"*******************************************************")
    print(f"Model")
    print(f"-------------------------------------------------------")
    print(f"Network             ==> UNet (ResNet Blocks)")
    print(f"Nº of parameters    ==> {n_parameters(model)}")
    print(f"Embedding           ==> Sinusoidal Positional Encoding")
    print(f"=======================================================")
else:
    pass

def sample_N(N, model, diffusion, img_size = img_size):
    if N == 1:
        xT = torch.randn(1, 1, img_size, img_size)
        x0 = diffusion.sample(xT, model)
        img = tensor_to_pil(x0)
        plt.imshow(img, cmap = "gray")
        plt.axis(False)
        plt.show()
    
    else:
        fig, axs = plt.subplots(1, N)
        for n in tqdm(range(N), desc = "Sampling"):
            xT = xT = torch.randn(1, 1, img_size, img_size)
            x0 = diffusion.sample(xT, model)
            img = tensor_to_pil(x0)
            axs[n].imshow(img, cmap = "gray")
            axs[n].axis(False)
        
        plt.show()

sample_N(1, model, diff_model)
