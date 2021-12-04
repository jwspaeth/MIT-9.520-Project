import time

import hydra
import torch
from torch import nn
from typing import *

class CDF(nn.Module):
    """
    Equivalent to a 2D convolution without weight sharing.
    """

    def __init__(self,
        in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels # Assume out_channels is the same as in_channels for now

        """
        Create weights.
        These are the biases for each sigmoid activation, and so are shape (out_channels).
        """
        self.weights = nn.Parameter(torch.rand((self.out_channels)))

    def forward(self, x):
        """
        x should be an image batch of shape (batch_size, channels, height, width)
        """
        logits = []
        for i in range(self.weights.shape[0]):
            logits.append(x + self.weights[i])
        logits = torch.stack(logits, dim=4)
        activations = nn.functional.sigmoid(logits)
        cdf = activations.sum(dim=1).permute((0,3,1,2))

        return cdf