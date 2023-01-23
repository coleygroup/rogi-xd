import logging
from typing import Iterable, Optional

import pandas as pd
from tdc import Oracle
from tdc.generation import MolGen

from pcmr.data.base import DataModule

logger = logging.getLogger(__name__)


class GuacaMolDataModule(DataModule):
    """The :class:`GuacaMolDataModule` computes Guacamol oracle values for 10,000 molecules
    randomly sampled from ZINC.

    NOTE: It is _not_ possible to alter this number programatically
    """

    __seed = 42
    __smis = MolGen("ZINC").get_data().sample(10000, random_state=__seed).smiles.tolist()
    __ORACLE_NAMES = {
        "LOGP",
        "AMLODIPINE_MPO",
        "SCAFFOLD HOP",
        "MEDIAN 1",
        "RANOLAZINE_MPO",
        "ARIPIPRAZOLE_SIMILARITY",
        "FEXOFENADINE_MPO",
        "ZALEPLON_MPO",
        "VALSARTAN_SMARTS",
        "CELECOXIB_REDISCOVERY",
        "PERINDOPRIL_MPO",
        "OSIMERTINIB_MPO",
        "QED",
    }
    __cache = {}

    @classmethod
    def seed(cls, seed: int):
        logger.info(f"Reseeding {cls.__name__}...")
        cls.__seed = seed
        cls.__smis = MolGen("ZINC").get_data().sample(10000, random_state=seed).smiles.tolist()
        cls.__cache = {}

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls.__ORACLE_NAMES

    @classmethod
    def get_tasks(cls, dataset: str) -> Iterable[str]:
        cls.check_dataset(dataset)

        return []

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        cls.check_dataset(dataset)
        if task is not None:
            raise ValueError(f"Invalid task! got: {task}. expected `None`")

        df = cls.__cache.get(dataset.upper())
        if df is not None:
            return df

        oracle = Oracle(dataset)
        y = oracle(cls.__smis)

        df = pd.DataFrame(dict(smiles=cls.__smis, y=y))
        cls.__cache[dataset.upper()] = df

        return df

    @classmethod
    def check_dataset(cls, dataset):
        if dataset.upper() not in cls.datasets:
            raise ValueError(f"Invalid dataset! got: '{dataset}'. expected one of {cls.datasets}.")
