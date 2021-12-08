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
        bias: bool=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

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
        #return self.effective_height-self.kernel_size+1

    @property
    def effective_width(self):
        return self.in_width+(self.padding*2)

    @property
    def output_width(self):
        return int((self.effective_width-self.kernel_size)/self.stride + 1)
        #return self.effective_width-self.kernel_size+1

    def forward(self, x):
        """
        x should be an image batch of shape (batch_size, channels, height, width)
        """

        start_time = time.time()
        # Apply padding to input
        padded_x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Build circulant matrix
        circulant = []
        for oc in range(self.out_channels):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    # Create kernel vector
                    current_kernel = self.weights[h, w, oc]
                    kernel_image = nn.functional.pad(current_kernel,
                        (w*self.stride, self.effective_width-self.kernel_size-(w*self.stride),
                            h*self.stride, self.effective_height-self.kernel_size-(h*self.stride)))
                    kernel_vector = torch.flatten(kernel_image)
                    circulant.append(kernel_vector)
        circulant = torch.stack(circulant) # (circulant_size, image_size)
        #print(f"Elapsed time to build circulant: {time.time()-start_time}")

        start_time = time.time()
        # Matmul circulant with input image
        padded_x_vector = torch.flatten(padded_x, start_dim=1) # (batch_size, image_size)
        circulant_transpose = circulant.transpose(0,1) # (image_size, circulant_size)
        matrix_sum = torch.matmul(padded_x_vector, circulant_transpose) # (batch_size, circulant_size)

        # Reshape
        output = matrix_sum.reshape((matrix_sum.shape[0], self.out_channels, self.output_height, self.output_width)) # (batch_size, out_channels, height, width)

        # Add bias
        if self.use_bias:
            output = output + self.bias.unsqueeze(dim=0)
        #print(f"Elapsed time to matrix multiply: {time.time()-start_time}")
        #breakpoint()

        return output