
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
import torch
from torch import nn

from omegaconf import OmegaConf


class DenseNet(LightningModule):

    def __init__(self, in_dim, hidden_dims, out_dim, optimizer_cfg=None):
        super().__init__()
        self.save_hyperparameters()

        self.in_dim = in_dim
        self.hidden_dims = OmegaConf.to_container(hidden_dims)
        self.out_dim = out_dim
        self.optimizer_cfg = optimizer_cfg

        self.layer_sizes = [self.in_dim] + self.hidden_dims + [self.out_dim]
        self.layers = []
        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

        self.net = nn.Sequential(*self.layers)
        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return instantiate(self.optimizer_cfg, params=self.parameters())

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        logits = self.net(x_flat)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        logits = self.forward(x)
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
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        argmaxes = logits.argmax(dim=1)
        hits = []
        for i in range(x.shape[0]):
            if argmaxes[i] == y[i]:
                hits.append(1)
            else:
                hits.append(0)
        hits = torch.tensor(hits, dtype=torch.float32)

        self.log("val_loss", loss)
        return {"loss": loss, "hits": hits}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output["loss"] for output in outputs]).mean()
        avg_acc = torch.cat([output["hits"] for output in outputs]).mean()
        print(f"Avg val acc: {avg_acc}")
        self.logger.experiment.add_scalar("val_loss_epoch", avg_loss, self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["target"]
        logits = self.forward(x)
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
