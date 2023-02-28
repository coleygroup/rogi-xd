import logging
from typing import Iterable, Optional, TypeVar
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchdrug.data
from torchdrug.layers import MLP
from torchdrug.tasks import PropertyPrediction

from ae_utils.char import LitCVAE, UnsupervisedDataset, SupervisedDataset
from ae_utils.supervisors import RegressionSupervisor
from ae_utils.utils.config import Configurable
from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.featurizers.mixins import BatchSizeMixin
from pcmr.models.gin import LitAttrMaskGIN, CustomDataset

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

ConfigurablePlModel = TypeVar("ConfigurablePlModel", pl.LightningModule, Configurable)


class LitFeaturizerMixin(BatchSizeMixin):
    DEFAULT_BATCH_SIZE = 256

    def __init__(
        self,
        model: ConfigurablePlModel,
        batch_size: Optional[int] = None,
        finetune_batch_size: Optional[int] = None,
        num_workers: int = 0,
        reinit: bool = False,
        **kwargs,
    ):
        if reinit:
            model = model.from_config(model.to_config())

        self.model = model
        self.batch_size = batch_size
        self.finetune_batch_size = finetune_batch_size or 64
        self.num_workers = num_workers


    @torch.inference_mode()
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        dataloader = self.build_unsupervised_loader(smis)

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            False,
            False,
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            enable_model_summary=False,
        )
        Xs = trainer.predict(self.model, dataloader)

        return torch.cat(Xs).numpy().astype(float)

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        self.setup_finetune()
        train_loader, val_loader = self.build_finetune_loaders(*splits)

        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            None,
            False,
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            max_epochs=20,
            enable_model_summary=False,
        )
        trainer.fit(self.model, train_loader, val_loader)

        return self


@FeaturizerRegistry.register("gin")
class GINFeaturizer(LitFeaturizerMixin, FeaturizerBase):
    def build_unsupervised_loader(self, smis):
        dataset = CustomDataset()
        dataset.load_smiles(smis, {}, atom_feature="pretrain", bond_feature="pretrain")

        return torchdrug.data.DataLoader(dataset, self.batch_size, num_workers=self.num_workers)

    def setup_finetune(self, output_dim: int = 1):
        self.model: LitAttrMaskGIN
        gin = self.model.task.model
        task = PropertyPrediction(gin, "y")
        task.mlp = MLP(gin.output_dim, [gin.output_dim] * (task.num_mlp_layer - 1) + [output_dim])

    def build_finetune_loaders(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]):
        if len(splits) == 1:
            smis, Y = splits[0]
            dset = CustomDataset()
            dset.load_smiles(smis, {"Y": Y})
            train, val = torch.utils.data.random_split(dset, [0.9, 0.1])
        elif 2 <= len(splits) <= 3:
            (smis_train, y_train), (smis_val, y_val), *_ = splits
            train = CustomDataset()
            train.load_smiles(smis_train, {"Y": y_train})
            val = CustomDataset()
            val.load_smiles(smis_val, {"Y": y_val})
        else:
            raise ValueError

        train_loader = torchdrug.data.DataLoader(
            train, self.batch_size, num_workers=self.num_workers
        )
        val_loader = torchdrug.data.DataLoader(val, self.batch_size, num_workers=self.num_workers)

        return train_loader, val_loader


@FeaturizerRegistry.register("vae")
class VAEFeaturizer(LitFeaturizerMixin, FeaturizerBase):
    def build_unsupervised_loader(self, smis):
        dset = UnsupervisedDataset(smis, self.model.tokenizer)

        return torch.utils.data.DataLoader(
            dset, self.batch_size, num_workers=self.num_workers, collate_fn=dset.collate_fn
        )

    def setup_finetune(self, output_dim: int = 1):
        self.model: LitCVAE
        self.model.supervisor = RegressionSupervisor(self.model.d_z, output_dim)
        self.model.v_sup = 1

    def build_finetune_loaders(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]):
        if len(splits) == 1:
            smis, Y = splits[0]
            dset = UnsupervisedDataset(smis, self.model.tokenizer)
            dset = SupervisedDataset(dset, Y)
            train, val, _ = torch.utils.data.random_split(dset, [0.8, 0.1, 0.1])
        elif 2 <= len(splits) <= 3:
            (smis_train, y_train), (smis_val, y_val), *_ = splits
            train = UnsupervisedDataset(smis_train, self.model.tokenizer)
            train = SupervisedDataset(train, y_train)
            val = UnsupervisedDataset(smis_val, self.model.tokenizer)
            val = SupervisedDataset(val, y_val)
        else:
            raise ValueError

        train_loader = torch.utils.data.DataLoader(
            train,
            self.finetune_batch_size,
            num_workers=self.num_workers,
            collate_fn=SupervisedDataset.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val,
            self.finetune_batch_size,
            num_workers=self.num_workers,
            collate_fn=SupervisedDataset.collate_fn,
        )

        return train_loader, val_loader
