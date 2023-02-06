from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchdrug.models import GIN
from torchdrug.layers import MLP, functional
from torchdrug.data import constant
from torchdrug.tasks import AttributeMasking

from pcmr.models.utils import PlMixin


class LitAttrMaskGIN(pl.LightningModule, PlMixin):
    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_h: Optional[list[int]] = None,
        gin_kwargs: Optional[Mapping[str, Any]] = None,
        view: str = "atom",
        mask_rate: float = 0.15,
        lr: float = 3e-4,
    ):
        super().__init__()

        d_h = d_h or [300, 300, 300, 300, 300]
        gin_kwargs = gin_kwargs or dict(batch_norm=True, readout="mean")
        gin = GIN(d_v, d_h, d_e, **gin_kwargs)
        task = AttributeMasking(gin, mask_rate)
        self.task = self.connect_task(task, view, gin)
        self.lr = lr

    def connect_task(self, task, view, model):
        task.view = view
        d_o = model.node_output_dim if hasattr(model, "node_output_dim") else model.output_dim
        n_label = constant.NUM_ATOM if view == "atom" else constant.NUM_AMINO_ACID
        task.mlp = MLP(d_o, [d_o] * (task.num_mlp_layer - 1) + [n_label])

        return task

    def forward(self, graph) -> Tensor:
        n_nodes = graph.num_nodes if self.task.view in ["atom", "node"] else graph.num_residues
        n_nodes_cum = n_nodes.cumsum(0)
        n_samples = (n_nodes * self.mask_rate).long().clamp(1)
        total_samples = n_samples.sum()
        sample2graph = functional._size_to_index(n_samples)
        node_idxs = (torch.rand(total_samples, device=self.device) * n_nodes[sample2graph]).long()
        node_idxs = node_idxs + (n_nodes_cum - n_nodes)[sample2graph]

        if self.task.view == "atom":
            inputs = graph.node_feature.float()
            inputs[node_idxs] = 0
        else:
            with graph.residue():
                graph.residue_feature[node_idxs] = 0
                graph.residue_type[node_idxs] = 0
            inputs = graph.residue_feature.float()

        output = self.task.model(graph, inputs)
        if self.task.view in ["node", "atom"]:
            X_v = output["node_feature"]
        else:
            X_v = output.get("residue_feature", output.get("node_feature"))

        return X_v[node_idxs]

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
        return self(batch["graph"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.task.parameters(), self.lr)
