import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Union

class CNN1d(LightningModule):
    """
    Convolutional neural network for classifying ESC50 data.
    Uses weight sharing.
    """

    def __init__(self,
        n_classes: int,
        in_channels: int,
        channels: List[int],
        local_kernel_size: int,
        hidden_dims: List[int],
        pool_kernel_size: int=2,
        padding: Union[int, List[int]]=0,
        bias: bool=True,
        batchnorm: bool=False,
        verbose: bool=False,
        optimizer_cfg: Optional[OmegaConf]=None):

        super().__init__()
        self.save_hyperparameters()

        assert len(channels) == 3
        assert len(hidden_dims) == 2
        assert type(padding) == int or (type(padding) == list and len(padding) == 3)

        self.n_classes = n_classes
        self.in_channels = in_channels
        self.channels = channels
        self.local_kernel_size = local_kernel_size
        self.hidden_dims = hidden_dims
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        if type(padding) is int:
            padding = [padding]*3
        self.bias = bias
        self.batchnorm = batchnorm
        self.verbose = verbose
        self.optimizer_cfg = optimizer_cfg

        # Declare layers
        self.local1 = nn.Conv1d(in_channels=in_channels, out_channels=channels[0], kernel_size=local_kernel_size, # 1
            padding=padding[0], bias=bias)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.batch1 = nn.BatchNorm1d(num_features=channels[0])
        self.local2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=local_kernel_size,
            padding=padding[0], bias=bias)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.batch2 = nn.BatchNorm1d(num_features=channels[1])
        self.local3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=local_kernel_size,
            padding=padding[0], bias=bias)
        self.pool3 = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.batch3 = nn.BatchNorm1d(num_features=channels[2])
        self.dense1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dense2 = nn.Linear(hidden_dims[1], n_classes)

        # Gather in list
        if self.batchnorm:
            self.module_list = []
            self.module_list += [self.local1, nn.ReLU(), self.pool1, self.batch1, self.local2, nn.ReLU(), self.pool2, self.batch2]
            self.module_list += [self.local3, nn.ReLU(), self.pool3, self.batch3, nn.Flatten(start_dim=1)]
            self.module_list += [self.dense1, nn.ReLU(), self.dense2]
        else:
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
            if self.verbose: print(f"Layer: {layer} -- Out shape: {out.shape}")
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

        argmaxes = logits.argmax(dim=1)
        hits = []
        for i in range(x.shape[0]):
            if argmaxes[i] == y[i]:
                hits.append(1)
            else:
                hits.append(0)
        hits = torch.tensor(hits, dtype=torch.float32)

        if self.trainer.global_step % 50 == 0:
            self.logger.experiment.add_scalar("train_loss_step", loss, self.trainer.global_step)

        return {"loss": loss, "hits": hits}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        avg_acc = torch.cat([output["hits"] for output in outputs]).mean()
        print(f"Avg train acc: {avg_acc}")
        self.logger.experiment.add_scalar("train_loss_epoch", avg_loss, self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        softmax, logits = self(x)
        loss = self.loss_fn(logits, y)

        argmaxes = logits.argmax(dim=1)
        hits = []
        for i in range(x.shape[0]):
            if argmaxes[i] == y[i]:
                hits.append(1)
            else:
                hits.append(0)
        hits = torch.tensor(hits, dtype=torch.float32)

        self.log("val_loss", self.loss_fn(logits, y))
        return {"loss": loss, "hits": hits}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        avg_acc = torch.cat([output["hits"] for output in outputs]).mean()
        print(f"Avg val acc: {avg_acc}")
        self.logger.experiment.add_scalar("val_loss_epoch", avg_loss, self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        softmax, logits = self(x)
        loss = self.loss_fn(logits, y)

        argmaxes = logits.argmax(dim=1)
        hits = []
        for i in range(x.shape[0]):
            if argmaxes[i] == y[i]:
                hits.append(1)
            else:
                hits.append(0)
        hits = torch.tensor(hits, dtype=torch.float32)

        self.log("test_loss", loss)
        return {"loss": loss, "hits": hits}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        avg_acc = torch.cat([output["hits"] for output in outputs]).mean()
        print(f"Avg test acc: {avg_acc}")
        self.logger.experiment.add_scalar("test_loss_epoch", avg_loss, self.trainer.current_epoch)