from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from pcmr.models.vae.tokenizer import Tokenizer


class UnsupervisedDataset(Dataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer):
        self.data = list(words)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Tensor:
        xs = self.tokenizer(self.data[i])

        return torch.tensor(xs, dtype=torch.long)

    @staticmethod
    def collate_fn(idss) -> list[Tensor]:
        return idss


class CachedUnsupervisedDataset(UnsupervisedDataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer, quiet: bool = False):
        self.data = [
            torch.tensor(tokenizer(w), dtype=torch.long)
            for w in tqdm(words, "Caching", unit="word", disable=quiet, leave=False)
        ]

    def __getitem__(self, i: int) -> Tensor:
        return self.data[i]


class SupervisedDataset(Dataset):
    def __init__(self, dset: UnsupervisedDataset, Y: np.ndarray):
        self.dset = dset
        self.Y = torch.from_numpy(Y).float()
    
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.dset[i], self.Y[i]
    
    @staticmethod
    def collate_fn(batch: Iterable[tuple[Tensor, Tensor]]) -> list[Tensor]:
        idss, ys = zip(*batch)

        return idss, torch.stack(ys)