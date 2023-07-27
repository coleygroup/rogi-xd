from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import Collection, Optional

import pandas as pd
from tqdm import tqdm

from rogi_xd.data import data
from rogi_xd.featurizers import FeaturizerBase
from rogi_xd.knn import rogi_knn
from rogi_xd.cli.rogi import RogiSubcommand
from rogi_xd.cli.utils.args import bounded, dataset_and_task
from rogi_xd.cli.utils.records import RogiKnnRecord

logger = logging.getLogger(__name__)


def _calc_rogi_knns(
    df: pd.DataFrame,
    dt_string: str,
    f: FeaturizerBase,
    n: int,
    repeats: Optional[int],
    ks: Collection[int],
) -> list[RogiKnnRecord]:
    """Calculate the ROGI-KNN of a given dataset

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe containing at least two comlums
        
        * `smiles` the SMILES string of each molecule in the dataset
        * `y`: the target value of each molecule in the dataset
    dt_string : str
        the string identifier of the dataset
    f : FeaturizerBase
        the featurizer with which to calculate feature representations of the molecules in the
        dataset
    n : int
        the maximum number of rows to take from a dataset. Larger datasets will be subsampled to `n`
    repeats : Optional[int]
        the number of repeats to use when subsampling. If subsampling isn't necessary, then will be
        set to 1
    ks : Collection[int], default=None
        the values of `k` to use. If empty, use all values of `k`

    Returns
    -------
    list[RogiKnnRecord]
        a list of length `repeats` containing one `RogiKnnRecord` for each repeat
    """
    if len(ks) == 0:
        ks = range(1, len(df) + 1)

    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")

        records = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            xs = f(df_sample.smiles.tolist())
            y = df_sample.y.values
            for k in tqdm(ks, "k"):
                result = rogi_knn(xs, y, k=k)
                record = RogiKnnRecord(str(f), dt_string, k, result)
                records.append(record)
    else:
        xs = f(df.smiles.tolist())

        records = []
        for k in tqdm(ks, "k"):
            result = rogi_knn(xs, df.y.values, k=k)
            records.append(RogiKnnRecord(str(f), dt_string, k, result))

    return records


class KnnSubcommand(RogiSubcommand):
    COMMAND = "knn"
    HELP = "Calculate the ROGI of (featurizer, dataset) pairs"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        RogiSubcommand._add_common_rogi_args(parser)
        parser.add_argument(
            "-k",
            type=bounded(1)(int),
            nargs="*",
            default=[5],
            help="the number of neighbors to use"
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )

        print(args)

        args.output = args.output or Path(f"results/raw/knn/{args.featurizer}.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        f = RogiSubcommand.build_featurizer(
            args.featurizer,
            args.batch_size,
            args.model_dir,
            args.num_workers,
            args.reinit,
            args.length,
        )

        records = []
        try:
            for d, t in args.datasets_tasks:
                logger.info(f"running dataset/task={d}/{t}")
                df = data.get(d, t)
                dt_string = f"{d}/{t}" if t else d

                try:
                    records_ = _calc_rogi_knns(df, dt_string, f, args.N, args.repeats, args.k)
                    records.extend(records_)
                except FloatingPointError as e:
                    logger.error(f"ROGI calculation failed! dataset/task={d}/{t}. Skipping...")
                    logger.error(e)
        finally:
            df = pd.DataFrame(records)
            df.to_csv(args.output, index=False)
            logger.info(f"Saved output to '{args.output}'")

            print(df)
