from abc import abstractmethod
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchdrug.data

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.models.gin import LitAttrMaskGIN, CustomDataset
from pcmr.models.vae.data import UnsupervisedDataset


class LitFeaturizer(FeaturizerBase):
    def __init__(
        self, model: LitAttrMaskGIN, *args, batch_size: int = 256, num_workers: int = 0, **kwargs
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.inference_mode()
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        dataloader = self.build_dataloader(smis)

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(False, False, accelerator="gpu" if gpus else "cpu", devices=gpus or 1)
        Xs = trainer.predict(self.model, dataloader)

        return torch.cat(Xs).numpy().astype(float)

    @abstractmethod
    def build_dataloader(self, smis) -> torch.utils.data.DataLoader:
        pass


@FeaturizerRegistry.register("gin")
class GINFeaturizer(LitFeaturizer):
    def build_dataloader(self, smis):
        dataset = CustomDataset()
        dataset.load_smiles(smis, {}, atom_feature="pretrain", bond_feature="pretrain")

        return torchdrug.data.DataLoader(dataset, self.batch_size, num_workers=self.num_workers)


@FeaturizerRegistry.register("vae")
class VAEFeaturizer(LitFeaturizer):
    def build_dataloader(self, smis):
        dataset = UnsupervisedDataset(smis, self.model.tokenizer)

        return torch.utils.data.DataLoader(dataset, self.batch_size, num_workers=self.num_workers)
