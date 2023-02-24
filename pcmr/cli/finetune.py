from argparse import ArgumentParser, Namespace
from itertools import repeat
import logging
from os import PathLike
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sklearn.model_selection import KFold

from ae_utils.char import LitCVAE
from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, FeaturizerRegistry
from pcmr.models.gin.model import LitAttrMaskGIN
from pcmr.rogi import rogi
from pcmr.utils import Metric
from pcmr.cli.utils.command import Subcommand
from pcmr.cli.utils.args import dataset_and_task
from pcmr.cli.utils.records import RogiRecord

logger = logging.getLogger(__name__)


def _calc_rogi(f: FeaturizerBase, smis: Iterable[str], y: Iterable[float]):
    X = f(smis)
    score, _, n_valid = rogi(y, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)

    return score, n_valid


def calc_rogi(
    f: FeaturizerBase, dataset: str, task: Optional[str], n: int, repeats: int
) -> list[RogiRecord]:
    df = data.get_all_data(dataset, task)

    dt_string = f"{dataset}/{task}" if task else dataset
    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")
        results = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            score, n_valid = _calc_rogi(f, df_sample.smiles.tolist(), df_sample.y.tolist())
            results.append(RogiRecord(f.alias, dt_string, n_valid, score))
    else:
        score, n_valid = _calc_rogi(f, df.smiles.tolist(), df.y.tolist())
        result = RogiRecord(f.alias, dt_string, n_valid, score)
        results = list(repeat(result, repeats))

    return results


class FinetuneSubcommand(Subcommand):
    COMMAND = "finetune"
    HELP = "Calculate the ROGI of (featurizer, dataset) pair after first finetuning the featurizer on a given split"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-d",
            "--dataset-task",
            "--dt",
            "--dataset",
            type=dataset_and_task,
            # nargs="+",
        )
        parser.add_argument(
            "-f", "--featurizer", type=lambda s: s.lower(), choices=FeaturizerRegistry.keys()
        )
        parser.add_argument(
            "-k", "--num-folds", type=int, default=5, help="the number of folds in cross-validation"
        )
        parser.add_argument("-N", type=int, default=10000, help="the number of data to subsample")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="the to which results should be written. If unspecified, will write to 'results/raw/finetune/FEATURIZER.csv'",
        )
        parser.add_argument(
            "-m", "--model-dir", help="the directory of a saved model for VAE or GIN featurizers"
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            help="the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size",
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
        # if args.input:
        #     args.datasets_tasks.extend(
        #         [dataset_and_task(l) for l in args.input.read_text().splitlines()]
        #     )
        args.output = args.output or Path(f"results/raw/finetune/{args.featurizer}.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        dataset, task = args.dataset_task
        dt_string = f"{dataset}/{task}" if task else dataset
        logger.info(f"running dataset/task={dt_string}")

        df = data.get_all_data(dataset, task)
        if len(df) > args.N:
            logger.info(f"Subsampling {args.N} rows from dataset with {len(df)} rows")
            df = df.sample(args.N)

        smis = df.smiles.values
        y = df.y.values

        records = []
        kf = KFold(args.num_folds)
        for i, (train_idxs, _) in enumerate(kf.split(smis, y)):
            logger.info(f"FOLD {i}:")
            logger.debug("  Reloading featurizer")
            featurizer = FinetuneSubcommand.build_featurizer(
                args.featurizer, args.batch_size, args.model_dir, args.num_workers
            )

            smis_train = [smis[i] for i in train_idxs]
            y_train = y[train_idxs]
            featurizer = featurizer.finetune((smis_train, y_train))

            score, n_valid = _calc_rogi(featurizer, smis, y)
            records.append(RogiRecord(featurizer.alias, dt_string, n_valid, score))

            logger.info(f"  ROGI: {score:0.3f}")
            logger.info(f"  n_valid: {n_valid}")

        df = pd.DataFrame(records)
        print(df)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved output CSV to '{args.output}'")

    @staticmethod
    def build_featurizer(
        featurizer: str,
        batch_size: Optional[int] = None,
        model_dir: Optional[PathLike] = None,
        num_workers: int = 0,
    ) -> FeaturizerBase:
        featurizer_cls = FeaturizerRegistry[featurizer]
        if featurizer == "vae":
            model = LitCVAE.load(model_dir)
        elif featurizer == "gin":
            model = LitAttrMaskGIN.load(model_dir)
        elif featurizer in ("chemgpt", "chemberta"):
            model = None
        else:
            model = None

        f = featurizer_cls(model=model, batch_size=batch_size, num_workers=num_workers)
        return f
