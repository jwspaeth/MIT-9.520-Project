import numpy as np
import os
import random
import requests
import zipfile

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def download_data():
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    r = requests.get(url, allow_redirects=True)

    if not os.path.exists("data/ESC50"):
        os.mkdir("data/ESC50")

    open('data/ESC50/data.zip', 'wb').write(r.content)
    with zipfile.ZipFile('data/ESC50/data.zip', 'r') as zip_ref:
        zip_ref.extractall("data/ESC50/")


class ESC50Dataset(Dataset):
    
    def __init__(self, data_path, split=None):
        super().__init__()

        self.data_path = data_path
        self.split = split

    def __getitem__(self, ind):
        pass

    def __len__(self):
        pass


class ESC50DataModule(LightningDataModule):

    def __init__(self, data_path, batch_size, shuffle=False, num_workers=0):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train = ESC50Dataset(data_path=self.data_path, use_tensor=True, split="train")
            self.val = ESC50Dataset(data_path=self.data_path, use_tensor=True, split="val")

        if stage == "test" or stage is None:
            self.test = ESC50Dataset(data_path=self.data_path, use_tensor=True, split="test")

    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def collate_fn(self, batch):
        batch = pd.DataFrame(batch).to_dict(orient="list")
        batch["input"] = torch.stack(batch["input"], axis=0)
        batch["target"] = torch.LongTensor(batch["target"])
        return batch