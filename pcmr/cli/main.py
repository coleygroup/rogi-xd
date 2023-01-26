from itertools import repeat
import logging
from typing import Optional

import numpy as np
import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerRegistry
from pcmr.featurizers.base import FeaturizerBase
from pcmr.rogi import rogi
from pcmr.utils import Metric

from pcmr.cli.args import build_parser
from pcmr.cli.utils import RogiCalculationResult

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


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s", level=logging.INFO
    )

    parser = build_parser()
    args = parser.parse_args()
    logger.info(f"Running with args: {args}")

    rows = []
    for featurizer in args.featurizers:
        for dataset, task in args.datasets_tasks:
            results = calc_rogi(featurizer, dataset, task, args.N, args.batch_size, args.repeats)
            rows.extend(results)

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(args.output, index=False)

    logger.info(f"Saved output CSV to '{args.output}'")
