from typing import Optional

import pandas as pd
from tdc import Oracle
from tdc.generation import MolGen

from pcmr.data.base import DataModule


class GuacaMolDataModule(DataModule):
    _ZINC_SMIS = MolGen('ZINC').get_data().sample(10000).smiles.tolist()
    _ORACLE_NAMES = {'LOGP', 'AMLODIPINE_MPO', 'SCAFFOLD HOP', 'MEDIAN 1', 'RANOLAZINE_MPO', 'ARIPIPRAZOLE_SIMILARITY', 'FEXOFENADINE_MPO', 'ZALEPLON_MPO', 'VALSARTAN_SMARTS', 'CELECOXIB_REDISCOVERY', 'PERINDOPRIL_MPO', 'OSIMERTINIB_MPO', 'QED'}

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls._ORACLE_NAMES

    @classmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        return set()
    
    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        oracle = Oracle(dataset)
        y = oracle(cls._ZINC_SMIS)

        return pd.DataFrame(dict(smiles=cls._ZINC_SMIS, y=y))