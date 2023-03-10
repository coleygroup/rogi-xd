from argparse import ArgumentParser, Namespace
import logging
from os import PathLike
from pathlib import Path
from typing import Optional, Union
import numpy as np

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

from ae_utils.char import LitCVAE
from pcmr.data import data
from pcmr.featurizers import FeaturizerBase, FeaturizerRegistry, VAEFeaturizer
from pcmr.models.gin import LitAttrMaskGIN
from pcmr.rogi import rogi
from pcmr.utils import Metric

from pcmr.cli.utils.args import dataset_and_task
from pcmr.cli.utils.command import Subcommand
from pcmr.cli.utils.records import CrossValdiationResult, RogiRecord, RogiAndCrossValRecord

logger = logging.getLogger(__name__)

SEED = 42
SCORING = ("r2", "neg_mean_squared_error", "neg_mean_absolute_error")
MODELS = {
    "KNN": KNeighborsRegressor(),
    "PLS": PLSRegression(),
    "RF": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEED),
    "SVR": SVR(),
    "MLP": MLPRegressor(random_state=SEED),
}
N_JOBS = 8


def _calc_rogi(
    df: pd.DataFrame, dt_string: str, f: FeaturizerBase, n: int, repeats: Optional[int]
) -> list[RogiRecord]:
    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")

        records = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            xs = f(df_sample.smiles.tolist())
            y = df_sample.y.values
            rr = rogi(xs, y, metric=f.metric, min_dt=0.01)
            record = RogiRecord(f.alias, dt_string, rr)
            records.append(record)
    elif isinstance(f, VAEFeaturizer):  # VAEs embed inputs stochastically
        records = []
        for _ in range(repeats):
            xs = f(df.smiles.tolist())
            rr = rogi(xs, df.y.values, metric=f.metric, min_dt=0.01)
            records.append(RogiRecord(f.alias, dt_string, rr))
    else:
        xs = f(df.smiles.tolist())
        rr = rogi(xs, df.y.values, metric=f.metric, min_dt=0.01)
        record = RogiRecord(f.alias, dt_string, rr)
        records = [record for _ in range(repeats)]

    return records


def _calc_cv(
    df: pd.DataFrame,
    dt_string: str,
    f: FeaturizerBase,
    n: int,
    cv: Optional[KFold] = None,
    name2model: Optional[dict] = None,
) -> list[RogiAndCrossValRecord]:
    name2model = name2model or MODELS

    if len(df) > n:
        logger.info(f"Subsampling {n} rows from dataset (N={len(df)})")

        df = df.sample(n)

    xs = f(df.smiles.tolist())
    y = df.y.values
    X = xs if isinstance(xs, np.ndarray) else np.array(xs)
    
    cvrs = []
    for name, model in name2model.items():
        logger.info(f"  MODEL: {name}")
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, verbose=1, n_jobs=N_JOBS)
        r2, neg_mse, neg_mae = (scores[f"test_{k}"] for k in SCORING)
        r2 = r2.mean()
        rmse = np.sqrt(-neg_mse).mean()
        mae = -neg_mae.mean()
        cvrs.append(CrossValdiationResult(name, r2, rmse, mae))

    rr = rogi(xs, y, metric=f.metric, min_dt=0.01)
    records = [RogiAndCrossValRecord(f.alias, dt_string, rr, cvr) for cvr in cvrs]

    return records


def calc(
    f: FeaturizerBase,
    dataset: str,
    task: Optional[str],
    n: int,
    repeats: Optional[int] = 5,
    cv: Optional[KFold] = None,
) -> Union[list[RogiRecord], list[RogiAndCrossValRecord]]:
    df = data.get(dataset, task)
    dt_string = f"{dataset}/{task}" if task else dataset

    if cv is not None:
        records = _calc_cv(df, dt_string, f, n, cv)
    else:
        records = _calc_rogi(df, dt_string, f, n, repeats)

    return records


class RogiSubcommand(Subcommand):
    COMMAND = "rogi"
    HELP = "Calculate the ROGI of (featurizer, dataset) pairs"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        xor_group = parser.add_mutually_exclusive_group(required=True)
        xor_group.add_argument(
            "-i",
            "--input",
            type=Path,
            help="A plaintext file containing a dataset/task entry on each line. Mutually exclusive with the '--datasets-tasks' argument",
        )
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
            "-f", "--featurizer", type=lambda s: s.lower(), choices=FeaturizerRegistry.keys()
        )
        parser.add_argument("-r", "--repeats", type=int, default=5)
        parser.add_argument("-N", type=int, default=10000, help="the number of data to subsample")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="the to which results should be written. If unspecified, will write to 'results/raw/rogi/FEATURIZER.{csv,json}', depending on the output data ('.json' if '--cg' is present, '.csv' otherwise)",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            help="the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size",
        )
        parser.add_argument(
            "-m", "--model-dir", help="the directory of a saved model for VAE or GIN featurizers"
        )
        parser.add_argument(
            "-c",
            "--num-workers",
            type=int,
            default=0,
            help="the number of CPUs to parallelize data loading over, if possible",
        )
        parser.add_argument(
            "--coarse-grain",
            "--cg",
            action="store_true",
            help="whether to store the raw coarse-graining results.",
        )
        parser.add_argument(
            "-k",
            "--num-folds",
            "--cv",
            nargs="?",
            type=int,
            const=5,
            help="the number of folds to use in cross-validation. If this flag is present, then this script will run in cross-validation mode, otherwise it will just perform ROGI calculation. Adding only the flag (i.e., just '-k') corresponds to a default of 5 folds, but a specific number may be specified",
        )
        parser.add_argument(
            "--reinit",
            action="store_true",
            help="randomize the weights of a pretrained model before using it",
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )

        suffix = "json" if args.coarse_grain else "csv"
        args.output = args.output or Path(f"results/raw/rogi/{args.featurizer}.{suffix}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        f = RogiSubcommand.build_featurizer(
            args.featurizer, args.batch_size, args.model_dir, args.num_workers, args.reinit
        )
        cv = KFold(args.num_folds, shuffle=True, random_state=SEED) if args.num_folds else None

        records = []
        try:
            for d, t in args.datasets_tasks:
                logger.info(f"running dataset/task={d}/{t}")
                try:
                    records_ = calc(f, d, t, args.N, args.repeats, cv)
                    records.extend(records_)
                except FloatingPointError as e:
                    logger.error(f"ROGI calculation failed! dataset/task={d}/{t}. Skipping...")
                    logger.error(e)
        finally:
            df = pd.DataFrame(records)
            if not args.coarse_grain:
                df = df.drop(["thresholds", "cg_sds", "n_clusters"], axis=1)
            print(df)

            if args.coarse_grain:
                df.to_json(args.output, indent=2)
            else:
                df.to_csv(args.output, index=False)
            logger.info(f"Saved output to '{args.output}'")

    @staticmethod
    def build_featurizer(
        featurizer: str,
        batch_size: Optional[int] = None,
        model_dir: Optional[PathLike] = None,
        num_workers: int = 0,
        reinit: bool = False,
    ) -> FeaturizerBase:
        featurizer_cls = FeaturizerRegistry[featurizer]
        if featurizer == "vae":
            model = LitCVAE.load(model_dir)
        elif featurizer == "gin":
            model = LitAttrMaskGIN.load(model_dir)
            # model = LitAttrMaskGIN.from_config(model.to_config())
        elif featurizer in ("chemgpt", "chemberta"):
            model = None
        else:
            model = None

        return featurizer_cls(
            model=model, batch_size=batch_size, num_workers=num_workers, reinit=reinit
        )
