from typing import Any, Mapping, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tdc.generation import MolGen
import torch

from torchdrug.models import GIN
from torchdrug.layers import MLP
from torchdrug.core import Registry as R
from torchdrug.data import DataLoader, constant
from torchdrug.tasks import AttributeMasking
from torchdrug.data.dataset import MoleculeDataset


# @R.register("datasets.Custom")
# class CustomDataset(MoleculeDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


class LitAttrMaskGIN(pl.LightningModule):
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
        model = GIN(d_v, d_h, d_e, **gin_kwargs)
        task = AttributeMasking(model, mask_rate)

        self.task = self.connect_task(task, view, model)
        self.lr = lr

    def connect_task(self, task, view, model):
        task.view = view
        d_o = model.node_output_dim if hasattr(model, "node_output_dim") else model.output_dim
        n_label = constant.NUM_ATOM if view == "atom" else constant.NUM_AMINO_ACID
        task.mlp = MLP(d_o, [d_o] * (task.num_mlp_layer - 1) + [n_label])

        return task

    def training_step(self, batch, batch_idx):
        loss, metrics = self.task(batch)

        self._log_split("train", metrics)
        self.log("loss", loss, True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.task(batch)
        acc = metrics["accuracy"]

        self._log_split("val", {"loss": loss, "accuracy": acc}, batch_size=len(batch["graph"]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.task.parameters(), self.lr)

    def _log_split(self, split: str, d: Mapping[str, Any], *args, **kwargs):
        self.log_dict({f"{split}/{k}": v for k, v in d.items()}, *args, **kwargs)


# dataset = CustomDataset()
# dataset.load_smiles(smis, {}, lazy=False, atom_feature="pretrain", bond_feature="pretrain")
# n_train = int(0.8 * len(dataset))
# n_val = len(dataset) - n_train
# train_dset, val_dset = torch.utils.data.random_split(dataset, [n_train, n_val])

# model = LitAttrMaskGIN(dataset.node_feature_dim, dataset.edge_feature_dim)

# model_name = "gin_am"
# checkpoint = ModelCheckpoint(
#     dirpath=f"chkpts/{model_name}/{dataset_name}",
#     filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
#     monitor="val/loss",
#     auto_insert_metric_name=False
# )
# early_stopping = EarlyStopping("val/loss")

# trainer = pl.Trainer(
#     WandbLogger(project=f"{model_name}-{dataset_name}"),
#     callbacks=[checkpoint, early_stopping],
#     accelerator="gpu",
#     devices=1,
#     check_val_every_n_epoch=3,
# )

# batch_size = 256
# num_workers = 0
# train_loader = DataLoader(train_dset, batch_size, num_workers=num_workers)
# val_loader = DataLoader(val_dset, batch_size, num_workers=num_workers)

# trainer.fit(model, train_loader, val_loader)
# torch.save(model, f"gin-am_{dataset_name}.pt")
