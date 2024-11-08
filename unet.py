import torch as torch
from torch import nn
import torchvision as tv

class PositionalEmbedding():
    """
    Class to obtain the positional embedding vector for given t.
    """
    def __init__(self, dim, theta = torch.tensor(10000)):
        super().__init__()
        self.dim = dim
        self.theta = theta
    
    def forward(self, t):
        half_dim = self.dim//2
        embedding = torch.log(self.theta)/(half_dim)
        embedding = torch.exp(- torch.arange(half_dim) * embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim = -1)
        return embedding


class Block(nn.Module):
    """
    Blocks of the UNet. Consist in a double convolution, adding the time embedding after the first one
    """
    def __init__(
            self, 
            input_channels, 
            output_channels, 
            time_embedding_dim
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1)
        self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(time_embedding_dim, output_channels)
    
    def forward(self, x, time_embedding):
        time_embedding = self.mlp(time_embedding)
        time_embedding = time_embedding[(...,) + (None,) * 2] # b c -> b c 1 1

        x = self.conv1(x)
        x = self.norm(x)
        x = x + time_embedding
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            input_channels = 1,
            output_channels = 1,
            time_embedding_dim = 100
    ):
        super().__init__()

        # Prepare channels
        channels = [2**i for i in range(2, 5)]

        # Layers
        self.downs = nn.ModuleList()
        self.upstconvs = nn.ModuleList()
        self.upsblock = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        # Encoder
        for channel in channels:
            self.downs.append(Block(input_channels, channel, time_embedding_dim))
            input_channels = channel
        
        # Bottleneck
        self.bottleneck = Block(channels[-1], 2 * channels[-1], time_embedding_dim)

        # Decoder
        for channel in reversed(channels):
            self.upstconvs.append(nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = "nearest"),
                nn.Conv2d(2 * channel, channel, kernel_size = 3, padding = 1)
            ))
            self.upsblock.append(Block(2 * channel, channel, time_embedding_dim))

        # Last convolution
        self.last_conv = nn.Conv2d(channels[0], output_channels, kernel_size = 1, stride = 1, padding = 0)

        # time embedding
        self.embedding = PositionalEmbedding(dim = time_embedding_dim)
        self.emb_mlp = nn.Linear(time_embedding_dim, time_embedding_dim)

    def forward(self, x, time):
        embedding = self.emb_mlp(self.embedding.forward(time))
        residual = []
        for down in self.downs:
            x = down(x, embedding)
            residual.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, embedding)

        for tconv, block in zip(self.upstconvs, self.upsblock):
            x = tconv(x)
            x = torch.cat((x, residual.pop()), dim = 1)
            x = block(x, embedding)
        
        x = self.last_conv(x)

        return x
