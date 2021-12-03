import time

import hydra
import torch
from torch import nn
from typing import *

class LocallyConnected2d(nn.Module):
    """
    Equivalent to a 2D convolution without weight sharing.
    """

    def __init__(self,
        in_channels: int,
        out_channels: int,
        in_height: int,
        in_width: int,
        kernel_size: int,
        padding: int=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.padding = padding

        """
        Create weights.
        These are shape (output_height, output_width, out_channels, in_channels, kernel_size, kernel_size).
        This is the same as nn.Conv2d with the additional height and width dimensions, since
        weights are not shared.
        """
        self.weights = nn.Parameter(torch.rand((self.output_height, self.output_width,
            out_channels, in_channels, kernel_size, kernel_size)))

    @property
    def effective_height(self):
        return self.in_height+(self.padding*2)

    @property
    def output_height(self):
        return self.effective_height-self.kernel_size+1

    @property
    def effective_width(self):
        return self.in_width+(self.padding*2)

    @property
    def output_width(self):
        return self.effective_width-self.kernel_size+1

    def forward(self, x):
        """
        x should be an image batch of shape (batch_size, channels, height, width)
        """
        padded_x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        start_time = time.time()
        circulant = []
        for oc in range(self.out_channels):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    # Create kernel vector
                    current_kernel = self.weights[h, w, oc]
                    kernel_image = nn.functional.pad(current_kernel,
                        (h, self.effective_height-self.kernel_size-h,
                            w, self.effective_width-self.kernel_size-w))
                    kernel_vector = torch.flatten(kernel_image)
                    circulant.append(kernel_vector)
        circulant = torch.stack(circulant)

        # Matmul circulant with input image
        padded_x_vector = torch.flatten(padded_x, start_dim=1)
        matrix_sum = torch.matmul(padded_x_vector, circulant.transpose(0, 1))

        # Reshape
        output = matrix_sum.reshape((matrix_sum.shape[0], self.out_channels, self.output_height, self.output_width))

        return output