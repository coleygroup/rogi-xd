import logging
from typing import Optional

import pandas as pd
from tdc import Oracle
from tdc.generation import MolGen

from rogi_xd.utils.utils import CACHE_DIR
from rogi_xd.data.base import DataModule

logger = logging.getLogger(__name__)


class GuacaMolDataModule(DataModule):
    """The :class:`GuacaMolDataModule` computes Guacamol oracle values for 10,000 molecules
    randomly sampled from ZINC.

    NOTE: It is _not_ possible to alter this number programatically
    """

    __seed = 42
    __smis = MolGen("ZINC", CACHE_DIR).get_data().sample(10000, random_state=__seed).smiles.tolist()
    __ORACLE_NAMES = {
        "SCAFFOLD HOP",
        "MEDIAN 1",
        "ARIPIPRAZOLE_SIMILARITY",
        "ZALEPLON_MPO",
        "CELECOXIB_REDISCOVERY",
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
    def tasks(cls, dataset: str) -> set[str]:
        cls.check_dataset(dataset)

        return []

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        cls.check_task(dataset, task)

        y = cls.__cache.get(dataset.upper())
        if y is None:
            oracle = Oracle(dataset)
            y = oracle(cls.__smis)
            cls.__cache[dataset.upper()] = y

        return pd.DataFrame(dict(smiles=cls.__smis, y=y))
