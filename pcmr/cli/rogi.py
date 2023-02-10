from argparse import ArgumentParser, Namespace
from itertools import repeat
import logging
from os import PathLike
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, FeaturizerRegistry
from pcmr.models.gin.model import LitAttrMaskGIN
from pcmr.models.vae.model import LitVAE
from pcmr.rogi import rogi
from pcmr.utils import Metric
from pcmr.cli.command import Subcommand
from pcmr.cli.utils import RogiCalculationRecord, dataset_and_task

logger = logging.getLogger(__name__)


def _calc_rogi(f: FeaturizerBase, smis: Iterable[str], y: Iterable[float]):
    X = f(smis)
    score, _ = rogi(y, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
    n_valid = len(X) - np.isnan(X).any(1).sum(0)

    return score, n_valid


def calc_rogi(
    f: FeaturizerBase, dataset: str, task: Optional[str], n: int, repeats: int
) -> list[RogiCalculationRecord]:
    df = data.get_all_data(dataset, task)

    dt_string = f"{dataset}/{task}" if task else dataset
    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")
        results = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            score, n_valid = _calc_rogi(f, df_sample.smiles.tolist(), df_sample.y.tolist())
            results.append(RogiCalculationRecord(f.alias, dt_string, n_valid, score))
    else:
        score, n_valid = _calc_rogi(f, df.smiles.tolist(), df.y.tolist())
        result = RogiCalculationRecord(f.alias, dt_string, n_valid, score)
        results = list(repeat(result, repeats))

    return results


class RogiSubcommand(Subcommand):
    COMMAND = "rogi"
    HELP = "Calculate the ROGI of (featurizer, dataset) pairs"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        xor_group = parser.add_mutually_exclusive_group(required=True)
        xor_group.add_argument("-i", "--input", type=Path, help="A plaintext file containing a dataset/task entry on each line. Mutually exclusive with the '--datasets-tasks' argument")
        xor_group.add_argument(
            "-d",
            "--datasets-tasks",
            "--dt",
            "--datasets",
            type=dataset_and_task,
            nargs="+",
            default=list(),
        )
        parser.add_argument(
            "-f",
            "--featurizer",
            type=lambda s: s.lower(),
            choices=FeaturizerRegistry.keys(),
        )
        parser.add_argument("-r", "--repeats", type=int, default=1)
        parser.add_argument("-N", type=int, default=10000, help="the number of data to sumbsample")
        parser.add_argument("-o", "--output", type=Path, help="the to which results should be written. If unspecified, will write to 'results/raw/FEATURIZEER.csv'")
        parser.add_argument(
            "-m", "--model-dir", help="the directory of a saved model for VAE or GIN featurizers"
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            help="the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size",
        )
        parser.add_argument("-c", "--num-workers", type=int, default=0, help="the number of CPUs to parallelize data loading over, if possible.")

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )
        args.output = args.output or Path(f"results/raw/{args.featurizer}.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        f = RogiSubcommand.build_featurizer(
            args.featurizer, args.batch_size, args.model_dir, args.num_workers
        )

        rows = []
        for d, t in args.datasets_tasks:
            logger.info(f"running dataset/task={d}/{t}")
            try:
                results = calc_rogi(f, d, t, args.N, args.repeats)
                rows.extend(results)
            except FloatingPointError as e:
                logger.error(f"ROGI calculation failed! dataset/task={d}/{t}. Skipping...")
                logger.error(e)
            except KeyboardInterrupt:
                logger.error("Interrupt detected! Exiting...")
                break

        df = pd.DataFrame(rows)
        print(df)
        df.to_csv(args.output, index=False)

        logger.info(f"Saved output CSV to '{args.output}'")

    @staticmethod
    def build_featurizer(
        featurizer: str,
        batch_size: Optional[int] = None,
        model_dir: Optional[PathLike] = None,
        num_workers: int = 0
    ) -> FeaturizerBase:
        featurizer_cls = FeaturizerRegistry[featurizer]
        if featurizer == "vae":
            model = LitVAE.load(model_dir)
        elif featurizer == "gin":
            model = LitAttrMaskGIN.load(model_dir)
        elif featurizer in ("chemgpt", "chemberta"):
            model = None
        else:
            model = None

        f = featurizer_cls(model=model, batch_size=batch_size, num_workers=num_workers)
        return f
