from argparse import ArgumentParser, Namespace
from enum import auto
import logging
from os import PathLike
from pathlib import Path
import pdb
from random import choices
from typing import Callable, Optional


import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tdc.generation import MolGen
from tdc.utils import fuzzy_search
from tdc.metadata import single_molecule_dataset_names as DATASETS
import torch

from torchdrug.core import Registry as R
from torchdrug.data import DataLoader
from torchdrug.data.dataset import MoleculeDataset

from pcmr.utils import AutoName
from pcmr.models.gin.model import LitAttrMaskGIN
from pcmr.cli.command import Subcommand

logger = logging.getLogger(__name__)


class ModelType(AutoName):
    GIN = auto()
    VAE = auto()


@R.register("datasets.Custom")
class CustomDataset(MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def fuzzy_lookup(choices: list[str]) -> Callable[[str], str]:
    def fun(choice: str):
        try:
            return fuzzy_search(choice, choices)
        except ValueError:
            return ValueError(f"Invalid choice! '{choice}' is not in possible choices: {choices}")

    return fun


class TrainSubcommand(Subcommand):
    COMMAND = "train"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("-o", "--output", help="where to save")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-i",
            "--input",
            type=Path,
            help="a plaintext file containing one SMILES string per line. Mutually exclusive with the '--dataset' argument.",
        )
        group.add_argument(
            "-d",
            "--dataset",
            type=fuzzy_lookup(DATASETS),
            choices=DATASETS,
            help="the TDC molecule generation dataset to train on. For more details, see https://tdcommons.ai/generation_tasks/molgen. Mutually exclusive with the '--input' argument",
        )
        parser.add_argument(
            "-N",
            type=int,
            help="the number of SMILES strings to subsample from _all_ supplied SMILES strings (i.e., from _both_ input sources)",
        )
        parser.add_argument("-m", "--model", type=ModelType.get, choices=list(ModelType))

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            smis = args.input.read_text().splitlines()
        else:
            smis = [MolGen(args.dataset).get_data().smiles.tolist()]

        if args.N:
            smis = choices(smis, k=args.N)

        if len(smis) == 0:
            raise ValueError("No smiles strings were supplied!")

        if args.model == ModelType.GIN:
            output = TrainSubcommand.train_gin(smis, args.input or args.dataset, args.output)
        elif args.model == ModelType.VAE:
            pass
        else:
            raise RuntimeError("Help! I've fallen and I can't get up! << CALL LIFEALERT >>")

        logger.info(f"Saved {args.model} model to {output}")

    @staticmethod
    def train_gin(smis: str, dataset_name: str, output: Optional[PathLike]):
        MODEL_NAME = "GIN"

        dataset = CustomDataset()
        dataset.load_smiles(smis, {}, lazy=True, atom_feature="pretrain", bond_feature="pretrain")

        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_dset, val_dset = torch.utils.data.random_split(dataset, [n_train, n_val])

        model = LitAttrMaskGIN(dataset.node_feature_dim, dataset.edge_feature_dim)

        checkpoint = ModelCheckpoint(
            dirpath=f"chkpts/{MODEL_NAME}/{dataset_name}",
            filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
            monitor="val/loss",
            auto_insert_metric_name=False,
        )
        early_stopping = EarlyStopping("val/loss")

        trainer = pl.Trainer(
            WandbLogger(project=f"{MODEL_NAME}-{dataset_name}"),
            callbacks=[checkpoint, early_stopping],
            accelerator="gpu",
            devices=1,
            check_val_every_n_epoch=3,
        )

        batch_size = 256
        num_workers = 0
        train_loader = DataLoader(train_dset, batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_dset, batch_size, num_workers=num_workers)

        trainer.fit(model, train_loader, val_loader)
        output = output or f"models/{MODEL_NAME}/{dataset_name}.pt"
        torch.save(model.state_dict(), output)
