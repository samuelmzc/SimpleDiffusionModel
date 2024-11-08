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

# Hyperparameters
batch_size, T, epochs, learning_rate = hyperparams()

# Import dataset and loaders
dataset, dataloader = set_and_loader(batch_size)

# Import the model
model = UNet()
diffusion_model = diffusion(T)
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)


def diff_loss(model, x0, t):
    x_new, noise = diffusion_model.forward_step(x0, t)
    noise_pred = model.forward(x0, t)
    return torch.nn.functional.l1_loss(noise, noise_pred)

print("Training.....")
J_train = []
for epoch in range(epochs):
    epoch_loss = 0
    for idx, image in enumerate(dataloader):
        optimizer.zero_grad()
        actual_batchsize = image.shape[0]
        t = torch.randint(0, T, (actual_batchsize,))
        x_next, noise = diffusion_model.forward_step(image, t)
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