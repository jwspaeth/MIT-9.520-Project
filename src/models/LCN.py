import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Union

from src.layers import LocallyConnected2d

class LCN(LightningModule):
    """
    Locally connected neural network.
    Does not use weight sharing.
    """

    def __init__(self,
        n_classes: int,
        channels: List[int],
        local_kernel_size: int,
        in_heights: List[int],
        in_widths: List[int],
        hidden_dims: List[int],
        pool_kernel_size: int=2,
        padding: Union[int, List[int]]=0,
        verbose: bool=False,
        optimizer_cfg: Optional[OmegaConf]=None):

        super().__init__()
        self.save_hyperparameters()

        assert len(channels) == 3
        assert len(in_heights) == 3
        assert len(in_widths) == 3
        assert len(hidden_dims) == 2
        assert type(padding) == int or (type(padding) == list and len(padding) == 3)

        self.n_classes = n_classes
        self.channels = channels
        self.in_heights = in_heights
        self.in_widths = in_widths
        self.local_kernel_size = local_kernel_size
        self.hidden_dims = hidden_dims
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        if type(padding) is int:
            padding = [padding]*3
        self.verbose = verbose
        self.optimizer_cfg = optimizer_cfg

        # Declare layers
        self.local1 = LocallyConnected2d(in_channels=3, out_channels=channels[0],
            in_height=in_heights[0], in_width=in_widths[0], kernel_size=local_kernel_size,
            padding=padding[0])
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.local2 = LocallyConnected2d(in_channels=channels[0], out_channels=channels[1],
            in_height=in_heights[1], in_width=in_widths[1], kernel_size=local_kernel_size,
            padding=padding[1])
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.local3 = LocallyConnected2d(in_channels=channels[1], out_channels=channels[2],
            in_height=in_heights[2], in_width=in_widths[2], kernel_size=local_kernel_size,
            padding=padding[2])
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.dense1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dense2 = nn.Linear(hidden_dims[1], n_classes)

        # Gather in list
        self.module_list = []
        self.module_list += [self.local1, nn.ReLU(), self.pool1, self.local2, nn.ReLU(), self.pool2]
        self.module_list += [self.local3, nn.ReLU(), self.pool3, nn.Flatten(start_dim=1)]
        self.module_list += [self.dense1, nn.ReLU(), self.dense2]

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.module_list[0](x)
        if self.verbose: print(f"Out shape: {out.shape}")
        for i, layer in enumerate(self.module_list[1:]):
            out = layer(out)
            if self.verbose: print(f"Layer: {layer}")
            if self.verbose: print(f"Out shape: {out.shape}")
        logits = out

        softmax = self.softmax(logits)
        return softmax, logits

    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            raise Exception("optimizer_cfg not defined in this model. Check the config file.")

        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        softmax, logits = self(x)
        loss = self.loss_fn(logits, y)
        if self.trainer.global_step % 50 == 0:
            self.logger.experiment.add_scalar("train_loss_step", loss, self.trainer.global_step)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.logger.experiment.add_scalar("train_loss_epoch", avg_loss, self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        softmax, logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", self.loss_fn(logits, y))
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.logger.experiment.add_scalar("val_loss_epoch", avg_loss, self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        softmax, logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.logger.experiment.add_scalar("test_loss_epoch", avg_loss, self.trainer.current_epoch)