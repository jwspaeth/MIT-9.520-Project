import os
import requests
import zipfile

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def download_data():
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    r = requests.get(url, allow_redirects=True)

    if not os.path.exists("data/ESC50"):
        os.mkdir("data/ESC50")

    open('data/ESC50/data.zip', 'wb').write(r.content)
    with zipfile.ZipFile('data/ESC50/data.zip', 'r') as zip_ref:
        zip_ref.extractall("data/ESC50/")


class ESC50Dataset(Dataset):
    
    def __init__(self, root, split=None, verbose=False):
        super().__init__()

        self.root = root
        if self.root[-1] != "/":
            self.root = self.root + "/"
        self.split = split
        self.verbose = verbose

        self.index = pd.read_csv(root+"/meta/esc50.csv")

        if split is not None:
            if split == "train":
                self.index = self.index[self.index.fold.isin([1,2,3])]
            elif split == "val":
                self.index = self.index[self.index.fold.isin([4])]
            elif split == "test":
                self.index = self.index[self.index.fold.isin([5])]

        self.data = []
        if self.verbose:
            loop = tqdm(range(len(self.index)))
        else:
            loop = range(len(self.index))
        for i in loop:
            filename = self.index.iloc[i].filename
            rate, sample = wavfile.read(self.root+"audio/"+filename)
            breakpoint()
            target = self.index.iloc[i].target
            self.data.append({"input": np.expand_dims(sample, axis=0).astype(np.float32), "target": target})

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class ESC50DataModule(LightningDataModule):

    def __init__(self, root, batch_size, shuffle=False, num_workers=0, verbose=False):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.verbose = verbose

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train = ESC50Dataset(root=self.root, split="train", verbose=self.verbose)
            self.val = ESC50Dataset(root=self.root, split="val", verbose=self.verbose)

        if stage == "test" or stage is None:
            self.test = ESC50Dataset(root=self.root, split="test", verbose=self.verbose)

    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def collate_fn(self, batch):
        batch = pd.DataFrame(batch).to_dict(orient="list")
        batch["input"] = torch.from_numpy(np.stack(batch["input"], axis=0))
        batch["target"] = torch.LongTensor(batch["target"])
        return batch