import os
import pickle
import requests
import zipfile

import hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from skimage import color
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def wav_to_spectrogram(audio, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    """ Creates a spectrogram of a wav file. Provided by FSDD.
    :param audio_path: path of wav file
    :param save_path:  path of spectrogram to save
    :param spectrogram_dimensions: number of pixels the spectrogram should be. Defaults (64,64)
    :param noverlap: See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    :param cmap: the color scheme to use for the spectrogram. Defaults to 'gray_r'
    :return:
    """

    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return ax.specgram(audio, cmap=cmap, Fs=2, noverlap=noverlap)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def download_data():

    print("Create directory")
    if not os.path.exists("data/FSDD"):
        os.mkdir("data/FSDD")

    print("Save")
    ds = hub.load("hub://activeloop/spoken_mnist")

    data = []
    bins = {}
    for sample in tqdm(ds.pytorch()):
        # Convert spectrogram from rbga to grayscale
        sample["spectrograms"] = color.rgb2gray(sample["spectrograms"].squeeze()[:,:,:3])
        sample["spectrograms"] = torch.from_numpy(sample["spectrograms"]).unsqueeze(dim=0)

        target = sample["labels"].item()
        speaker = sample["speakers"][0]

        if target not in bins.keys():
            bins[target] = {}

        if speaker not in bins[target].keys():
            bins[target][speaker] = -1

        bins[target][speaker] += 1
        file_path = f"data/FSDD/{target}_{speaker}_{bins[target][speaker]}.pickle"
        print(f"Target: {target} -- Speaker: {speaker} -- Bin: {bins[target][speaker]}")
        with open(file_path, "wb") as file:
            pickle.dump(sample, file)


class FSDDDataset(Dataset):
    
    def __init__(self, root, pad=True, split=None, verbose=False):
        super().__init__()

        self.root = root
        if self.root[-1] != "/":
            self.root = self.root + "/"
        self.split = split
        self.verbose = verbose

        self.filenames = os.listdir(root)
        self.parse_file_ind(self.filenames[0])

        # Split filenames
        train_inds = list(range(10, 50))
        val_inds = list(range(5, 10))
        test_inds = list(range(0, 5))
        if split is not None:
            if split == "train":
                self.filenames = [filename for filename in self.filenames if self.parse_file_ind(filename) in train_inds]
            elif split == "val":
                self.filenames = [filename for filename in self.filenames if self.parse_file_ind(filename) in val_inds]
            elif split == "test":
                self.filenames = [filename for filename in self.filenames if self.parse_file_ind(filename) in test_inds]

        # Gather data
        self.data = []
        if self.verbose:
            loop = tqdm(range(len(self.filenames)))
        else:
            loop = range(len(self.filenames))
        for i in loop:
            filename = self.filenames[i]
            with open(self.root + filename, 'rb') as file:
                sample = pickle.load(file)
                self.data.append(sample)

        # # Pad audio
        # lengths = []
        # for sample in self.data:
        #     lengths.append(sample["audio"].shape[1])
        # max_length = max(lengths)


    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)

    def parse_file_ind(self, filename):
        body = filename.split(".")[0]
        ind = int(body.split("_")[-1])
        return ind


class FSDDDataModule(LightningDataModule):

    def __init__(self, root, batch_size, data_type="spectrograms", shuffle=False, num_workers=0, test_split="test",
        verbose=False):
        super().__init__()

        assert data_type in ["spectrograms"]

        self.root = root
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.verbose = verbose
        self.test_split = test_split

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train = FSDDDataset(root=self.root, split="train", verbose=self.verbose)
            self.val = FSDDDataset(root=self.root, split="val", verbose=self.verbose)

        if stage == "test" or stage is None:
            self.test = FSDDDataset(root=self.root, split=self.test_split, verbose=self.verbose)

    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

    def collate_fn(self, batch):
        batch = pd.DataFrame(batch).to_dict(orient="list")
        final_batch = {}
        final_batch["input"] = torch.cat(batch[self.data_type], axis=0).type(torch.float32)
        final_batch["target"] = torch.cat(batch["labels"]).squeeze()
        return final_batch