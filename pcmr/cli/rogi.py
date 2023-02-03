from argparse import ArgumentParser, FileType, Namespace
from itertools import repeat
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, FeaturizerRegistry
from pcmr.rogi import rogi
from pcmr.utils import Metric
from pcmr.cli.command import Subcommand
from pcmr.cli.utils import RogiCalculationResult, dataset_and_task

logger = logging.getLogger(__name__)


def _calc_rogi(f, smis: Iterable[str], y: Iterable[float]):
    X = f(smis)
    score, _ = rogi(y, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
    n_valid = len(X) - np.isnan(X).any(1).sum(0)

    return score, n_valid


def calc_rogi(
    f: FeaturizerBase, dataset: str, task: Optional[str], n: int, repeats: int
) -> list[RogiCalculationResult]:
    dt = f"{dataset}/{task}"
    df = data.get_all_data(dataset, task)
    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")
        results = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            score, n_valid = _calc_rogi(f, df_sample.smiles.tolist(), df_sample.y.tolist())
            results.append(RogiCalculationResult(f, dt, n_valid, score))
    else:
        score, n_valid = _calc_rogi(f, df.smiles.tolist(), df.y.tolist())
        result = RogiCalculationResult(f, dt, n_valid, score)
        results = list(repeat(result, repeats))

    return results


class RogiSubcommand(Subcommand):
    COMMAND = "rogi"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("-i", "--input", type=FileType("r"))
        parser.add_argument(
            "--datasets-tasks",
            "--dt",
            "--datasets",
            type=dataset_and_task,
            nargs="+",
            default=list(),
        )
        parser.add_argument(
            "-f",
            "--featurizers",
            type=lambda s: s.lower(),
            nargs="+",
            choices=FeaturizerRegistry.keys(),
        )
        parser.add_argument("-r", "--repeats", type=int, default=1)
        parser.add_argument("-N", type=int, default=10000, help="the number of data to sumbsample")
        parser.add_argument("-o", "--output", type=Path)
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            help="the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size",
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input is not None:
            args.datasets_tasks.extend([dataset_and_task(line.strip()) for line in args.input])

        rows = []
        for f in args.featurizers:
            f = FeaturizerRegistry[f](args.batch_size)
            for d, t in args.datasets_tasks:
                logger.info(f"running dataset/task={d}/{t}, features={f}")
                try:
                    results = calc_rogi(f, d, t, args.N, args.repeats)
                    rows.extend(results)
                except FloatingPointError as e:
                    logger.error(
                        f"ROGI calculation failed! dataset/task={d}/{t}, features={f}. Skipping..."
                    )
                    logger.error(e)

        df = pd.DataFrame(rows)
        print(df)
        df.to_csv(args.output, index=False)

        logger.info(f"Saved output CSV to '{args.output}'")
