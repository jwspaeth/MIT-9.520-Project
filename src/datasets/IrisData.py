import numpy as np
import random

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class IrisDataset(Dataset):
    
    def __init__(self, data_path, split=None):
        super().__init__()

        self.data_path = data_path
        self.split = split

        self.enum = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }

        self.train_split = .8
        self.data = pd.read_csv(self.data_path)
        random_inds = list(range(len(self.data)))
        random.shuffle(random_inds)
        n_train = int(len(self.data)*.8)
        n_test = len(self.data) - n_train
        self.train = self.data.iloc[random_inds[:n_train]]
        self.test = self.data.iloc[random_inds[n_train:]]

        if self.split == "train":
            self.data = self.train
        elif self.split == "test":
            self.data = self.test

    def __getitem__(self, ind):
        sample = self.data.iloc[ind]
        x = np.array(sample[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], dtype=np.float32)
        y = self.enum[sample["Species"]]
        return {"input": x, "target": y}

    def __len__(self):
        return len(self.data)


class IrisDataModule(LightningDataModule):

    def __init__(self, data_path, batch_size, shuffle=False):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        """No val necessary for toy problem"""
        self.train = IrisDataset(data_path=self.data_path, split="train")
        self.test = IrisDataset(data_path=self.data_path, split="test")
        print(f"Train length: {len(self.train)}")
        print(f"Test length: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.test, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    def collate_fn(self, batch):
        batch = pd.DataFrame(batch).to_dict(orient="list")
        batch["input"] = np.stack(batch["input"], axis=0)
        batch["input"] = torch.from_numpy(batch["input"])
        batch["target"] = torch.LongTensor(batch["target"])
        return batch




