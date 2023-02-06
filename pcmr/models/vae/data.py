from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from pcmr.models.vae.tokenizer import Tokenizer


class UnsupervisedDataset(Dataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer):
        self.data = list(words)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Tensor:
        s = self.tokenizer(self.data[i])
        ids = self.tokenizer.tokens2ids(s)

        return torch.tensor(ids, dtype=torch.long)

    def to_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(idss) -> list[Tensor]:
        return idss


class CachedUnsupervisedDataset(UnsupervisedDataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer):
        self.data = [torch.tensor(tokenizer(w), dtype=torch.long) for w in words]

    def __getitem__(self, i: int) -> Tensor:
        return self.data[i]
