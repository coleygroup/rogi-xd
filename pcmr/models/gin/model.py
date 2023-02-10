from __future__ import annotations

from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchdrug.models import GIN
from torchdrug.layers import MLP
from torchdrug.data import constant
from torchdrug.tasks import AttributeMasking
from torchdrug.data import feature

from pcmr.models.mixins import LoggingMixin, SaveAndLoadMixin
from pcmr.utils.utils import Configurable

DEFAULT_ATOM_DIM = sum(len(v) for v in [feature.atom_vocab, feature.chiral_tag_vocab]) + 1
DEFAULT_BOND_DIM = sum(len(v) for v in [feature.bond_type_vocab, feature.bond_dir_vocab])


class LitAttrMaskGIN(pl.LightningModule, Configurable, LoggingMixin, SaveAndLoadMixin):
    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_DIM,
        d_e: int = DEFAULT_BOND_DIM,
        d_h: Optional[list[int]] = None,
        gin_kwargs: Optional[Mapping[str, Any]] = None,
        mask_rate: float = 0.15,
        lr: float = 3e-4,
    ):
        super().__init__()

        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h or [300, 300, 300, 300, 300]
        self.gin_kwargs = gin_kwargs or dict(batch_norm=True, readout="mean")
        model = GIN(self.d_v, self.d_h, self.d_e, **self.gin_kwargs)
        task = AttributeMasking(model, mask_rate)
        self.task = self.connect_task(task, model)
        self.lr = lr

    def connect_task(self, task, model):
        task.view = "atom"
        d_o = model.node_output_dim if hasattr(model, "node_output_dim") else model.output_dim
        n_label = constant.NUM_ATOM
        task.mlp = MLP(d_o, [d_o] * (task.num_mlp_layer - 1) + [n_label])

        return task

    def forward(self, batch) -> Tensor:
        graph = batch["graph"]
        output = self.task.model(graph, graph.node_feature.float())
        X = output["graph_feature"]

        return X

    def training_step(self, batch, batch_idx):
        loss, metrics = self.task(batch)

        self._log_split("train", metrics)
        self.log("loss", loss, True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.task(batch)
        acc = metrics["accuracy"]

        self._log_split("val", {"loss": loss, "accuracy": acc}, batch_size=len(batch["graph"]))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.task.parameters(), self.lr)

    def to_config(self) -> dict:
        return {
            "d_v": self.d_v,
            "d_e": self.d_e,
            "d_h": self.d_h,
            "gin_kwargs": self.gin_kwargs,
            "mask_rate": self.task.mask_rate,
            "lr": self.lr,
        }

    @classmethod
    def from_config(cls, config: dict) -> LitAttrMaskGIN:
        return cls(**config)
