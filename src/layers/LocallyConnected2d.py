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
        stride: int=1,
        padding: int=0,
        bias: bool=True,
        verbose: bool=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.verbose = verbose

        """
        Create weights.
        These are shape (output_height, output_width, out_channels, in_channels, kernel_size, kernel_size).
        This is the same as nn.Conv2d with the additional height and width dimensions, since
        weights are not shared.
        """
        self.weights = nn.Parameter(torch.rand((self.output_height, self.output_width,
            out_channels, in_channels, kernel_size, kernel_size)))

        if self.use_bias:
            self.bias = nn.Parameter(torch.rand((out_channels, self.output_height, self.output_width)))

    @property
    def effective_height(self):
        return self.in_height+(self.padding*2)

    @property
    def output_height(self):
        return int((self.effective_height-self.kernel_size)/self.stride + 1)

    @property
    def effective_width(self):
        return self.in_width+(self.padding*2)

    @property
    def output_width(self):
        return int((self.effective_width-self.kernel_size)/self.stride + 1)

    def forward(self, x):
        """
        Avoids sparse multiplications by doing dot product between kernels and image patches.

        x should be an image batch of shape (batch_size, channels, height, width)
        """

        start_time = time.time()
        # Apply padding to input
        padded_x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        start_time = time.time()
        # Build patches
        patches = []
        for h in range(self.output_height):
            for w in range(self.output_width):
                current_patch = padded_x[:,:,h*self.stride:(h*self.stride)+self.kernel_size, w*self.stride:(w*self.stride)+self.kernel_size]
                patches.append(current_patch)
        patches = torch.stack(patches, dim=1) # Shape (batch_size, output_height*output_width, in_channels, kernel_size, kernel_size)
        patches = torch.repeat_interleave(patches, self.out_channels, dim=1) # Shape (batch_size, output_height*output_width*out_channels, in_channels, kernel_size, kernel_size)
        patches = torch.flatten(patches, start_dim=2) # Shape (batch_size, output_height*output_width*out_channels, in_channels*kernel_size*kernel_size)

        # Build kernels
        kernels = torch.flatten(self.weights, end_dim=2) # Shape (output_height*output_width*out_channels, in_channels, kernel_size, kernel_size)
        kernels = torch.flatten(kernels, start_dim=1) # Shape (output_height*utput_width*out_channels, in_channels*kernel_size*kernel_size)
        if self.verbose: print(f"Elapsed time for build: {time.time()-start_time}")

        start_time = time.time()
        # Batched dot product
        matrix_mul = kernels * patches
        matrix_sum = torch.sum(matrix_mul, dim=2)

        # Reshape
        output = matrix_sum.reshape((x.shape[0], self.out_channels, self.output_height, self.output_width)) # (batch_size, out_channels, height, width)

        # Add bias
        if self.use_bias:
            output = output + self.bias.unsqueeze(dim=0)
        if self.verbose: print(f"Elapsed time for compute: {time.time()-start_time}")

        return output