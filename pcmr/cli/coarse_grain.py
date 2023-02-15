from argparse import Namespace
from itertools import repeat
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, VAEFeaturizer
from pcmr.rogi import rogi
from pcmr.utils import Metric
from pcmr.cli.rogi import RogiSubcommand
from pcmr.cli.utils import CoarseGrainCalculationRecord, dataset_and_task

logger = logging.getLogger(__name__)


def _calc_cg(f: FeaturizerBase, smis: Iterable[str], y: Iterable[float]):
    X = f(smis)
    score, _, n_valid, (thresholds, sds) = rogi(
        y, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01, return_cg=True
    )

    return n_valid, thresholds, sds


def calc_cg(
    f: FeaturizerBase, dataset: str, task: Optional[str], n: int, repeats: int
) -> list[CoarseGrainCalculationRecord]:
    df = data.get_all_data(dataset, task)

    dt_string = f"{dataset}/{task}" if task else dataset

    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")
        records = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            n_valid, thresholds, sds = _calc_cg(f, df_sample.smiles.tolist(), df_sample.y.tolist())
            record = CoarseGrainCalculationRecord(f.alias, dt_string, n_valid, thresholds, sds)
            records.append(record)
    elif isinstance(f, VAEFeaturizer):  # VAEs embed inputs stochastically
        records = []
        for _ in range(repeats):
            n_valid, thresholds, sds = _calc_cg(f, df.smiles.tolist(), df.y.tolist())
            record = CoarseGrainCalculationRecord(f.alias, dt_string, n_valid, thresholds, sds)
            records.append(record)

    else:
        n_valid, thresholds, sds = _calc_cg(f, df.smiles.tolist(), df.y.tolist())
        record = CoarseGrainCalculationRecord(f.alias, dt_string, n_valid, thresholds, sds)
        records = list(repeat(record, repeats))

    return records


class CoarseGrainSubcommand(RogiSubcommand):
    COMMAND = "cg"
    HELP = "Calculate just the coarse-grained property distribution of a (featurizer, dataset) pair"

    @staticmethod
    def func(args: Namespace):
        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )
        args.output = args.output or Path(f"results/raw/cg/{args.featurizer}.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        f = CoarseGrainSubcommand.build_featurizer(
            args.featurizer, args.batch_size, args.model_dir, args.num_workers
        )

        rows = []
        try:
            for d, t in args.datasets_tasks:
                logger.info(f"running dataset/task={d}/{t}")
                try:
                    results = calc_cg(f, d, t, args.N, args.repeats)
                    rows.extend(results)
                except FloatingPointError as e:
                    logger.error(f"ROGI calculation failed! dataset/task={d}/{t}. Skipping...")
                    logger.error(e)
                except KeyboardInterrupt:
                    logger.error("Interrupt detected! Exiting...")
                    break
        finally:
            df = pd.DataFrame(rows)
            print(df)
            # df.to_json()
            df.to_json(args.output.with_suffix(".json"), indent=2)
            logger.info(f"Saved output CSV to '{args.output}'")
