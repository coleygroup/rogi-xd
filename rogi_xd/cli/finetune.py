from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import Iterable

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.utils.data

from ae_utils.char import LitCVAE, UnsupervisedDataset, SemisupervisedDataset
from ae_utils.char.tokenizer import Tokenizer
from ae_utils.supervisors import RegressionSupervisor, ContrastiveSupervisor

from rogi_xd.data import data
from rogi_xd.featurizers import VAEFeaturizer
from rogi_xd.cli.utils.command import Subcommand
from rogi_xd.cli.utils.args import bounded, dataset_and_task
from rogi_xd.cli.rogi import _calc_rogi

logger = logging.getLogger(__name__)


def finetune_vae(
    model: LitCVAE,
    smis: Iterable[str],
    Y: np.ndarray,
    dt_string: str,
    v_sup: float = 50,
    batch_size: int = 64,
    num_workers: int = 0
):
    MODEL_NAME = "vae"

    model.supervisor = ContrastiveSupervisor(None, None)
    model.v_sup = v_sup
    train_loader, val_loader = build_dataloaders(smis, Y, model.tokenizer, batch_size, num_workers)

    pl_logger = WandbLogger(project=f"{MODEL_NAME}_{dt_string}_finetune")
    early_stopping = EarlyStopping("val/sup", patience=5)

    trainer = pl.Trainer(
        pl_logger, False, [early_stopping], accelerator="gpu", devices=1, max_epochs=100
    )
    trainer.fit(model, train_loader, val_loader)

    pl_logger.save()
    return model


def setup_supervision(model: LitCVAE, v_sup: float = 50):
    model.supervisor = ContrastiveSupervisor()
    model.v_sup = 50
    
    return model


def build_dataloaders(
    smis: Iterable[str], Y: np.ndarray, tokenizer: Tokenizer, batch_size: int, num_workers: int = 0
):
    dset = UnsupervisedDataset(smis, tokenizer)
    dset = SemisupervisedDataset(dset, Y)
    train, val, _ = torch.utils.data.random_split(dset, [0.8, 0.1, 0.1])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size,
        num_workers=num_workers,
        collate_fn=SemisupervisedDataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size,
        num_workers=num_workers,
        collate_fn=SemisupervisedDataset.collate_fn,
    )

    return train_loader, val_loader


class FinetuneSubcommand(Subcommand):
    COMMAND = "finetune"
    HELP = "Calculate the ROGI of (VAE, dataset) pair after first finetuning the VAE on a given 80%% split of the dataset"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-d",
            "--dataset-task",
            "--dt",
            "--dataset",
            type=dataset_and_task,
        )
        parser.add_argument(
            "-r", "--repeats", type=int, default=5, help="the number of repeats to perform"
        )
        parser.add_argument("-f", type=bounded(0, 1)(float), help="the fraction of the dataset on which to finetune the VAE.")
        parser.add_argument("-N", type=int, default=10000, help="the number of data to subsample")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="the to which results should be written. If unspecified, will write to 'results/raw/finetune/FEATURIZER.csv'",
        )
        parser.add_argument(
            "-m", "--model-dir", help="the directory of a saved VAE model"
        )
        parser.add_argument("--v-sup", type=float, default=50)
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=64,
            help="the batch size to use in when finetuning the VAE",
        )
        parser.add_argument(
            "-c",
            "--num-workers",
            type=int,
            default=0,
            help="the number of CPUs to parallelize data loading over, if possible.",
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        args.output = args.output or Path(f"results/raw/finetune/vae.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        dataset, task = args.dataset_task
        dt_string = f"{dataset}/{task}" if task else dataset
        logger.info(f"running dataset/task={dt_string}")

        df = data.get_all_data(dataset, task)
        smis = df.smiles.values
        Y = df.y.values

        N = len(df)
        k = int(args.f * N)

        records = []
        for i in range(args.repeats):
            logger.info(f"REPEAT {i}:")
            logger.debug("  Reloading featurizer")
            model = LitCVAE.load(args.model_dir)

            idxs = np.random.choice(N, k, replace=False)
            smis_train = [smis[i] for i in idxs]
            Y_train = Y[idxs]

            model = finetune_vae(
                model, smis_train, Y_train, dt_string, args.v_sup, args.batch_size, args.num_workers
            )
            featurizer = VAEFeaturizer(model, num_workers=args.num_workers)

            records_ = _calc_rogi(df, dt_string, featurizer, args.N, args.repeats, True)
            records.extend(records_)

        df = pd.DataFrame(records).drop(["thresholds", "cg_sds", "n_clusters"], axis=1)
        print(df)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved output CSV to '{args.output}'")
