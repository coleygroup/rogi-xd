from argparse import ArgumentError, ArgumentParser, Action, Namespace
from dataclasses import dataclass, asdict
from itertools import repeat
import logging
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerRegistry
from pcmr.featurizers.base import FeaturizerBase
from pcmr.rogi import rogi
from pcmr.utils import Metric


class DatasetAndTaskAction(Action):
    def __init__(
        self,
        *args,
        choices: Iterable,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._choices = set(choices)

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence],
        option_string: Optional[str] = None
    ):
        if isinstance(values, str):
            dataset, task = self.parse_value(values)
            setattr(namespace, self.dest, (dataset, task))
        else:
            datasets_tasks = [self.parse_value(value) for value in values]
            setattr(namespace, self.dest, datasets_tasks)

    def parse_value(self, value: str):
        tokens = value.split('/')
        if len(tokens) == 1:
            dataset = tokens[0]
            task = None
        elif len(tokens) == 2:
            dataset, task = tokens
        else:
            raise ArgumentError(self, f"arguments must of form: ('A', 'A/B')")

        dataset = dataset.upper()
        task = task or None
        if self._choices and dataset not in self._choices:
            raise ArgumentError(self, f"invalid choice: 'dataset' must one of {self._choices}")
        return dataset,task


# @dataclass
class RogiCalculationResult(NamedTuple):
    featurizer: str
    dataset: str
    task: Optional[str]
    n_valid: int
    rogi: float


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--featurizers",
        type=lambda s: s.lower(), nargs="+", choices=FeaturizerRegistry.keys()
    )
    parser.add_argument(
        "--datasets-tasks", "--dt", "--datasets",
        action=DatasetAndTaskAction,
        nargs="+",
        choices=data.datasets,
    )
    parser.add_argument("-r", "--repeats", type=int, default=1)
    parser.add_argument("-N", type=int, default=10000, help="the number of data to sumbsample")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    return parser


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


def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s:%(name)s - %(message)s', level=logging.INFO
    )

    parser = build_parser()
    args = parser.parse_args()
    print(args)

    rows = []
    for featurizer in args.featurizers:
        for dataset, task in args.datasets_tasks:
            results = calc_rogi(featurizer, dataset, task, args.N, args.batch_size, args.repeats)
            rows.extend(results)

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(args.output, index=False)
