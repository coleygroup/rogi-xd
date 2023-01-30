from argparse import ArgumentParser, Namespace
from itertools import repeat
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, FeaturizerRegistry
from pcmr.rogi import rogi
from pcmr.utils import Metric

from pcmr.cli.command import Subcommand
from pcmr.cli.utils import RogiCalculationResult, DatasetAndTaskAction

logger = logging.getLogger(__name__)


def calc_rogi(
    featurizer: str, dataset: str, task: Optional[str], N: int, batch_size: int, repeats: int
) -> list[RogiCalculationResult]:
    f: FeaturizerBase = FeaturizerRegistry[featurizer]()
    df = data.get_data(dataset, task, N)

    X = f(df.smiles.tolist())
    score, _ = rogi(df.y.tolist(), True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
    n_valid = len(X) - np.isnan(X).any(1).sum(0)

    result = RogiCalculationResult(featurizer, dataset, task, n_valid, score)
    if len(df) < N:
        results = list(repeat(result, repeats))
    else:
        results = [result]
        for _ in range(repeats - 1):
            df = data.get_data(dataset, task, N)

            X = f(df.smiles.tolist())
            score, _ = rogi(df.y.tolist(), True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
            n_valid = len(X) - np.isnan(X).any(1).sum(0)
            results.append(RogiCalculationResult(featurizer, dataset, task, n_valid, score))

    return results


class RogiSubcommand(Subcommand):
    COMMAND = "rogi"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-f",
            "--featurizers",
            type=lambda s: s.lower(),
            nargs="+",
            choices=FeaturizerRegistry.keys(),
        )
        parser.add_argument(
            "--datasets-tasks",
            "--dt",
            "--datasets",
            action=DatasetAndTaskAction,
            nargs="+",
            choices=data.datasets,
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
        rows = []
        for featurizer in args.featurizers:
            for dataset, task in args.datasets_tasks:
                results = calc_rogi(
                    featurizer, dataset, task, args.N, args.batch_size, args.repeats
                )
                rows.extend(results)

        df = pd.DataFrame(rows)
        print(df)
        df.to_csv(args.output, index=False)

        logger.info(f"Saved output CSV to '{args.output}'")
