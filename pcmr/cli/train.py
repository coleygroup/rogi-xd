from argparse import ArgumentParser, Namespace
from datetime import date
import logging
from os import PathLike
from pathlib import Path
from random import choices
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tdc.generation import MolGen
from tdc.metadata import single_molecule_dataset_names as DATASETS
from torch import nn
import torch.cuda
import torch.utils.data
from torchdrug.core import Registry as R
from torchdrug.data import DataLoader
from torchdrug.data.dataset import MoleculeDataset

from pcmr.models.gin import LitAttrMaskGIN
from pcmr.models.vae import LitVAE, CharDecoder, CharEncoder, Tokenizer, CachedUnsupervisedDataset
from pcmr.cli.command import Subcommand
from pcmr.cli.utils import ModelType, bounded, fuzzy_lookup
from pcmr.models.vae.data import UnsupervisedDataset

logger = logging.getLogger(__name__)


@R.register("datasets.Custom")
class CustomDataset(MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "pretrain a VAE or GIN model via unsupervised learning"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("model", type=ModelType.get, choices=list(ModelType))
        parser.add_argument("-o", "--output", help="where to save")
        xor_group = parser.add_mutually_exclusive_group(required=True)
        xor_group.add_argument(
            "-i",
            "--input",
            type=Path,
            help="a plaintext file containing one SMILES string per line. Mutually exclusive with the '--dataset' argument.",
        )
        xor_group.add_argument(
            "-d",
            "--dataset",
            type=fuzzy_lookup(DATASETS),
            choices=DATASETS,
            help="the TDC molecule generation dataset to train on. For more details, see https://tdcommons.ai/generation_tasks/molgen. Mutually exclusive with the '--input' argument",
        )
        parser.add_argument(
            "-N",
            type=bounded(lo=1)(int),
            help="the number of SMILES strings to subsample. Must be >= 1",
        )
        parser.add_argument(
            "-c",
            "--num-workers",
            type=int,
            default=0,
            help="the number of workers to use for data loading",
        )
        parser.add_argument(
            "-g",
            "--gpus",
            type=int,
            help="the number of GPUs to use (if any). If unspecified, will use GPU if available",
        )
        parser.add_argument(
            "--chkpt",
            help="the path of a checkpoint file from a previous run from which to resume training",
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            smis = args.input.read_text().splitlines()
        else:
            smis = MolGen(args.dataset).get_data().smiles.tolist()

        if args.N:
            smis = choices(smis, k=args.N)

        if args.gpus is None:
            logger.debug("GPU unspecifeid... Will use GPU if available")
            args.gpus = 1 if torch.cuda.is_available() else 0

        if len(smis) == 0:
            raise ValueError("No smiles strings were supplied!")

        if args.model == ModelType.GIN:
            func = TrainSubcommand.train_gin
        elif args.model == ModelType.VAE:
            func = TrainSubcommand.train_vae
        else:
            raise RuntimeError("Help! I've fallen and I can't get up! << CALL LIFEALERT >>")

        model, output = func(
            smis,
            args.dataset or args.input.stem,
            args.output,
            args.num_workers,
            args.gpus,
            args.chkpt,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output)
        logger.info(f"Saved {args.model} model to {output}")

    @staticmethod
    def train_gin(
        smis: list[str],
        dataset_name: str,
        output: Optional[PathLike],
        num_workers: int = 0,
        gpus: Optional[int] = None,
        chkpt: Optional[PathLike] = None,
    ) -> tuple[pl.LightningModule, Path]:
        MODEL_NAME = "GIN"
        TODAY = date.today().isoformat()

        dataset = CustomDataset()
        dataset.load_smiles(smis, {}, lazy=True, atom_feature="pretrain", bond_feature="pretrain")

        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_dset, val_dset = torch.utils.data.random_split(dataset, [n_train, n_val])

        model = LitAttrMaskGIN(dataset.node_feature_dim, dataset.edge_feature_dim)
        checkpoint = ModelCheckpoint(
            dirpath=f"chkpts/{MODEL_NAME}/{dataset_name}/{TODAY}",
            filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
            monitor="val/loss",
            auto_insert_metric_name=False,
        )
        early_stopping = EarlyStopping("val/loss")

        trainer = pl.Trainer(
            WandbLogger(project=f"{MODEL_NAME}-{dataset_name}"),
            callbacks=[checkpoint, early_stopping],
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            check_val_every_n_epoch=3,
            max_epochs=100,
        )

        batch_size = 256
        train_loader = DataLoader(train_dset, batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_dset, batch_size, num_workers=num_workers)

        if chkpt:
            logger.info(f"Resuming training from checkpoint '{chkpt}'")

        trainer.fit(model, train_loader, val_loader, ckpt_path=chkpt)
        output = output or f"models/{MODEL_NAME}/{dataset_name}.pt"

        return model, Path(output)

    @staticmethod
    def train_vae(
        smis: list[str],
        dataset_name: str,
        output: Optional[PathLike],
        num_workers: int = 0,
        gpus: Optional[int] = None,
        chkpt: Optional[PathLike] = None,
    ):
        MODEL_NAME = "VAE"
        TODAY = date.today().isoformat()

        tokenizer = Tokenizer.smiles_tokenizer()
        embedding = nn.Embedding(len(tokenizer), 64, tokenizer.PAD)
        encoder = CharEncoder(embedding)
        decoder = CharDecoder(tokenizer, embedding)
        model = LitVAE(tokenizer, encoder, decoder)

        dataset = CachedUnsupervisedDataset(smis, tokenizer)
        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_dset, val_dset = torch.utils.data.random_split(dataset, [n_train, n_val])

        checkpoint = ModelCheckpoint(
            dirpath=f"chkpts/{MODEL_NAME}/{dataset_name}/{TODAY}",
            filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
            monitor="val/loss",
            auto_insert_metric_name=False,
        )
        early_stopping = EarlyStopping("val/loss")

        trainer = pl.Trainer(
            WandbLogger(project=f"{MODEL_NAME}-{dataset_name}"),
            callbacks=[checkpoint, early_stopping],
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            check_val_every_n_epoch=3,
            max_epochs=100,
        )

        batch_size = 256
        train_loader = DataLoader(
            train_dset,
            batch_size,
            num_workers=num_workers,
            collate_fn=UnsupervisedDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dset, batch_size, num_workers=num_workers, collate_fn=UnsupervisedDataset.collate_fn
        )

        if chkpt:
            logger.info(f"Resuming training from checkpoint '{chkpt}'")

        trainer.fit(model, train_loader, val_loader, ckpt_path=chkpt)
        output = output or f"models/{MODEL_NAME}/{dataset_name}.pt"

        return model, Path(output)
