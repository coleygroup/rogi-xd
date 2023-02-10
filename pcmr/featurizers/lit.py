from abc import abstractmethod
from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchdrug.data

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.models.gin import CustomDataset
from pcmr.models.vae.data import UnsupervisedDataset


class LitFeaturizer(FeaturizerBase):
    def __init__(
        self,
        model: pl.LightningModule,
        batch_size: Optional[int] = 256,
        num_workers: int = 0,
        **kwargs,
    ):
        self.model = model
        self.batch_size = batch_size or 256
        self.num_workers = num_workers

    @torch.inference_mode()
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        dataloader = self.build_dataloader_unsup(smis)

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(False, False, accelerator="gpu" if gpus else "cpu", devices=gpus or 1)
        Xs = trainer.predict(self.model, dataloader)

        return torch.cat(Xs).numpy().astype(float)

    def finetune(self, smis: Iterable[str], targets: ArrayLike) -> Self:
        targets = np.array(targets)
        self.setup_finetune()
        train_loader, val_loader = self.build_finetune_loaders(smis, targets)
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            None,
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            max_epochs=10,
        )
        trainer.fit(self.model, train_loader, val_loader)

        return self

    def build_finetune_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size,
            num_workers=self.num_workers,
            collate_fn=UnsupervisedDataset.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size,
            num_workers=self.num_workers,
            collate_fn=UnsupervisedDataset.collate_fn
        )
        
        return train_loader,val_loader
    
    @abstractmethod
    def build_dataloader_unsup(self, smis: list[str]) -> torch.utils.data.DataLoader:
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

        return torch.utils.data.DataLoader(
            dataset, self.batch_size, num_workers=self.num_workers, collate_fn=dataset.collate_fn
        )
