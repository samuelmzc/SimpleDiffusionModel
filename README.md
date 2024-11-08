# SimpleDiffusionModel
Implementation of basic diffusion model for image generation. The model is tested on the MNIST dataset, but only picking the handwritten numbers that correspond to number 1. It consist in 6 python files:

* **MNIST.py**: Creates the dataset class.
* **process_data.py**: Process the input data, and define other functions to analize results.
* **diffusion.py**: The diffusion model.
* **unet.py**: The UNet model used to train.
* **training.py**: Starts the training and saves the weights.
* **sampling.py**: Creates a noisy images and tries to create the handwritte 1.

This diffusion model is based on the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). For a given image, the model adds noise to it along an arbitrary number of time steps T. A UNet trains to predict the noise that will be added to the image at a certain time step t. Then, for the sampling, create a noisy image $\mathcal{N(0,I)}$, and run backwards through the time steps, substracting the noise the UNet predicted for each time step.

The result for this model is a noisy images. This may happen because the noise in the forward step is added linear. 
