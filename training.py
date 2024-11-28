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

# Fix the seed
torch.manual_seed(14573)

# Hyperparameters
batch_size, T, epochs, learning_rate, embedding_dim = hyperparams() 

# Import dataset and loaders
dataset, dataloader = set_and_loader(batch_size)
n_batchs = (len(dataset)//batch_size) + 1

# Import the model
model = UNet()
diffusion_model = diffusion_cosine(T)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)


def diff_loss(model, x0, t):
    x_new, noise = diffusion_model.forward(x0, t)
    noise_pred = model.forward(x_new, t)
    return torch.nn.functional.mse_loss(noise, noise_pred)

# print information
verbose = True

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


print("Training.....")
J_train = []
for epoch in range(epochs):
    epoch_loss = 0
    for idx, image in tqdm(enumerate(dataloader), total = n_batchs):
        optimizer.zero_grad()
        actual_batchsize = image.shape[0]
        t = torch.randint(0, T, (actual_batchsize,))
        loss = diff_loss(model, image, t)
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    epoch_loss /= len(dataset)
    J_train.append(epoch_loss.detach().numpy())
    print(f"Epoch {epoch + 1} | loss : {epoch_loss}")


# Save the weights
path = "weights/"
name = "UNet_diff.pth"
torch.save(model.state_dict(), path + name)

plot_cost(J_train, save = False)
