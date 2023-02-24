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
SCORING = 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
MODELS = {
    'KNN': KNeighborsRegressor(), 
    'PLS': PLSRegression(), 
    'RF': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEED), 
    'SVR': SVR(), 
    'MLP': MLPRegressor(random_state=SEED)
}


def _calc_cv(
    X: np.ndarray, y: np.ndarray, cv: KFold, name2model: dict = None
) -> list[CrossValdiationResult]:
    name2model = name2model or MODELS
    records = []
    
    for name, model in name2model.items():
        logger.info(f"  MODEL: {name}")
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, verbose=1)
        r2, neg_mse, neg_mae = (scores[f"test_{k}"] for k in SCORING)
        r2 = r2.mean()
        rmse = np.sqrt(-neg_mse).mean()
        mae = -neg_mae.mean()
        records.append(CrossValdiationResult(name, r2, rmse, mae))
    
    return records


def calc(
    f: FeaturizerBase,
    dataset: str,
    task: Optional[str],
    n: int,
    repeats: int,
    cv: Optional[KFold] = None
) -> Union[list[RogiRecord], list[RogiAndCrossValRecord]]:
    df = data.get_all_data(dataset, task)
    dt_string = f"{dataset}/{task}" if task else dataset

    if len(df) > n:
        logger.info(f"Repeating with {repeats} subsamples (n={n}) from dataset (N={len(df)})")

        records = []
        for _ in range(repeats):
            df_sample = df.sample(n)
            X = f(df_sample.smiles.tolist())
            y = df_sample.y.values
            rr = rogi(y, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
            if cv:
                cvrs = _calc_cv(X, y, cv)
                cv_recs = [RogiAndCrossValRecord(f.alias, dt_string, rr, cvr) for cvr in cvrs]
                records.extend(cv_recs)
            else:
                record = RogiRecord(f.alias, dt_string, rr)
                records.append(record)
                
    elif isinstance(f, VAEFeaturizer):  # VAEs embed inputs stochastically
        records = []
        for _ in range(repeats):
            X = f(df.smiles.tolist())
            rr = rogi(df.y.values, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
            if cv:
                cvrs = _calc_cv(X, y, cv)
                cv_recs = [RogiAndCrossValRecord(f.alias, dt_string, rr, cvr) for cvr in cvrs]
                records.extend(cv_recs)
            else:
                record = RogiRecord(f.alias, dt_string, rr)
                records.append(record)

    else:
        X = f(df.smiles.tolist())
        rr = rogi(df.y.values, True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
        if cv:
            cvrs = _calc_cv(X, y, cv)
            cv_recs = [RogiAndCrossValRecord(f.alias, dt_string, rr, cvr) for cvr in cvrs]
            records.extend(cv_recs)
        else:
            record = RogiRecord(f.alias, dt_string, rr)
            records = [record for _ in range(repeats)]

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
        parser.add_argument("-r", "--repeats", type=int, default=1)
        parser.add_argument("-N", type=int, default=10000, help="the number of data to subsample")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="the to which results should be written. If unspecified, will write to 'results/raw/rogi/FEATURIZER.csv'",
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
        parser.add_argument("--coarse-grain", "--cg", action="store_true")
        parser.add_argument("-k", "--num-folds", nargs="?", type=int, const=5)

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )

        suffix = ".json" if args.coarse_grain else ".csv"
        args.output = args.output or Path(f"results/raw/rogi/{args.featurizer}.{suffix}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        f = RogiSubcommand.build_featurizer(
            args.featurizer, args.batch_size, args.model_dir, args.num_workers
        )

        records = []
        try:
            for d, t in args.datasets_tasks:
                logger.info(f"running dataset/task={d}/{t}")
                try:
                    records = calc(f, d, t, args.N, args.repeats)
                    records.extend(records)
                except FloatingPointError as e:
                    logger.error(f"ROGI calculation failed! dataset/task={d}/{t}. Skipping...")
                    logger.error(e)
        finally:
            df = pd.DataFrame(records)
            if not args.coarse_grain:
                df = df.drop(["thresholds", "sds_cg"], axis=1)
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

        return featurizer_cls(model=model, batch_size=batch_size, num_workers=num_workers)
