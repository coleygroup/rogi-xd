import logging
from typing import Optional

import pandas as pd
from tdc import Oracle
from tdc.generation import MolGen

from pcmr.data.base import DataModule

logger = logging.getLogger(__name__)

class GuacaMolDataModule(DataModule):
    __SEED = 42
    __SMIS = MolGen('ZINC').get_data().sample(10000, random_state=__SEED).smiles.tolist()
    __ORACLE_NAMES = {'LOGP', 'AMLODIPINE_MPO', 'SCAFFOLD HOP', 'MEDIAN 1', 'RANOLAZINE_MPO', 'ARIPIPRAZOLE_SIMILARITY', 'FEXOFENADINE_MPO', 'ZALEPLON_MPO', 'VALSARTAN_SMARTS', 'CELECOXIB_REDISCOVERY', 'PERINDOPRIL_MPO', 'OSIMERTINIB_MPO', 'QED'}
    
    @classmethod
    def seed(cls, seed: int):
        logger.info(f"Reseeding {cls.__name__}...")
        cls.__SEED = seed

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls.__ORACLE_NAMES

    @classmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        return set()
    
    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        oracle = Oracle(dataset)
        y = oracle(cls.__SMIS)

        return pd.DataFrame(dict(smiles=cls.__SMIS, y=y))