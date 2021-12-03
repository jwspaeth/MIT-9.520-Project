import random

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

class CIFAR10Dataset(CIFAR10):

    def __init__(self, split, *args, use_tensor=False, **kwargs):

        random.seed(0)

        # Initialize transform
        transform = None
        if use_tensor:
            transform = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(dtype=torch.float32)
                ])

        # Pick split
        if split in ["train", "val"]:
            train = True
        elif split == "test":
            train = False

        # Initialize parent
        super().__init__(*args, **kwargs, train=train, transform=transform)

        # Random split train and val indices
        self.inds = list(range(super().__len__()))
        if split in ["train", "val"]:
            random.shuffle(self.inds)

            if split == "train":
                self.inds = self.inds[:int(.8*len(self.inds))]
            else:
                self.inds = self.inds[int(.8*len(self.inds)):]

        self.use_tensor = use_tensor

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, ind):
        sample = super().__getitem__(self.inds[ind])
        return {"input": sample[0], "target": sample[1]}


class CIFAR10DataModule(LightningDataModule):

    def __init__(self, root, batch_size, shuffle=False):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train = CIFAR10Dataset(root=self.root, use_tensor=True, split="train")
            self.val = CIFAR10Dataset(root=self.root, use_tensor=True, split="val")

        if stage == "test" or stage is None:
            self.test = CIFAR10Dataset(root=self.root, use_tensor=True, split="test")

    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def collate_fn(self, batch):
        batch = pd.DataFrame(batch).to_dict(orient="list")
        batch["input"] = torch.stack(batch["input"], axis=0)
        batch["target"] = torch.LongTensor(batch["target"])
        return batch
