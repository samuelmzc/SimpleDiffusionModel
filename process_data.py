import numpy as np
import matplotlib.pyplot as plt

import torch as torch
import torchvision as tv

from MNIST import MNIST_3s


def hyperparams():
    """
    Set hyperparameters of the model
    """

    batch_size = 128
    epochs = 20
    learning_rate = 1e-3
    T = 100
    embedding_dim = 32
    return batch_size, T, epochs, learning_rate, embedding_dim


def n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_and_loader(batch_size):
    """
    Data set and Data loader

    Arguments:
    batch_size -- int

    Returns:
    dataloader -- torch.utils.DataLoader
    """

    transformation = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda img: 2 * img - 1)
    ])

    dataset = MNIST_3s(transformation)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    return dataset, dataloader


def tensor_to_pil(tensor):
    """
    Apply transformations to return a tensor into an image

    Arguments:
    tensor -- torch.tensor

    Returns:
    image -- PIL.Image
    """

    inverse_transforms = tv.transforms.Compose([
        tv.transforms.Lambda(lambda img : img.permute(1, 2, 0)),
        tv.transforms.Lambda(lambda img : (img + 1) * 255/2), # un normalize
        tv.transforms.Lambda(lambda img : img.detach().numpy().astype(np.uint8)),
        tv.transforms.ToPILImage()
    ])

    if len(tensor.shape) == 4:
        tensor = tensor[0, :, :, :]

    image = inverse_transforms(tensor)
    return image


def show_samples(dataset):
    """
    Show 5 samples of the set

    Arguments:
    dataset -- torch.dataset
    """

    n_samples = 5
    fig, ax = plt.subplots(1, n_samples)
    for i , tensor in enumerate(dataset):
        image = tensor_to_pil(tensor)
        ax[i].imshow(image, cmap = "gray")
        ax[i].axis(False)
        if i == n_samples - 1:
            break
    plt.show()


def plot_cost(train_hist, dev_hist = None, save = True):
    """
    Plot the cost of the training set, and from test set if given.
    """
    epochs_int = len(train_hist)
    epochs = np.linspace(1, epochs_int, epochs_int)
    plt.plot(epochs, train_hist, label = "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel(r"MSE")
    plt.grid()
    plt.tight_layout()

    if dev_hist:
        plt.plot(epochs, dev_hist, label = "Dev cost")

    if save == True:
        plt.savefig(f"loss/{epochs_int}eps.pdf")
    else:
        pass

    plt.show()
