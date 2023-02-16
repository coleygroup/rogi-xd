from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

from pcmr.cli.rogi import RogiSubcommand
from pcmr.data import data
from pcmr.cli.utils import CrossValdiationRecord, dataset_and_task
from pcmr.featurizers.base import FeaturizerBase

logger = logging.getLogger(__name__)

SEED = 42
SCORING = 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'


def run_cv(
    dataset: str,
    task: Optional[str],
    N: int,
    name2model: dict,
    featurizer: FeaturizerBase,
    cv: KFold
) -> list[CrossValdiationRecord]:
    dt_string = f"{dataset}/{task}" if task else dataset
    logger.info(f"running dataset/task={dt_string}")

    df = data.get_all_data(dataset, task)
    if len(df) > N:
        logger.info(f"Subsampling {N} rows from dataset with {len(df)} rows")
        df = df.sample(N)

    X = featurizer(df.smiles.values.tolist())
    y = np.array(df.y.values)

    records = []
    for name, model in name2model.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, verbose=1)
        r2, neg_mse, neg_mae = (scores[f"test_{k}"] for k in SCORING)
        r2 = r2.mean()
        rmse = np.sqrt(-neg_mse).mean()
        mae = -neg_mae.mean()

        record = CrossValdiationRecord(featurizer.alias, dt_string, len(X), name, r2, rmse, mae)
        records.append(record)
    
    return records


class CrossValidateSubcommand(RogiSubcommand):
    COMMAND = "cv"
    HELP = "Calculate the cross-validated model error with a suite of models for a given (featurizer, dataset) pair."

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser = RogiSubcommand.add_args(parser)
        parser.add_argument("-k", "--num-folds", type=int, default=5)

        return parser
    
    @staticmethod
    def func(args: Namespace):
        MODELS = {
            'KNN': KNeighborsRegressor(), 
            'PLS': PLSRegression(2), 
            'RF': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEED), 
            'SVR': SVR(), 
            'MLP': MLPRegressor(random_state=SEED)
        }

        if args.input:
            args.datasets_tasks.extend(
                [dataset_and_task(l) for l in args.input.read_text().splitlines()]
            )
        args.output = args.output or Path(f"results/raw/cv/{args.featurizer}.csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)


        featurizer = RogiSubcommand.build_featurizer(
            args.featurizer, args.batch_size, args.model_dir, args.num_workers
        )
        cv = KFold(args.num_folds, shuffle=True, random_state=SEED)

        records = []
        for dataset, task in args.datasets_tasks:
            records.extend(run_cv(dataset, task, args.N, MODELS, featurizer, cv))

        df = pd.DataFrame(records)
        print(df)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved output CSV to '{args.output}'")

