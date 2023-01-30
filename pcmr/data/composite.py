from typing import Optional

import pandas as pd

from pcmr.data.base import DataModule
from pcmr.data.tdc import TdcDataModule
from pcmr.data.guacamol import GuacaMolDataModule
from pcmr.exceptions import InvalidDatasetError

# from pcmr.data.molnet import MoleculeNetDataModule


class CompositeDataModule(DataModule):
    __MODULES: list[DataModule] = [TdcDataModule, GuacaMolDataModule]
    __dset2module = {dset: module for module in __MODULES for dset in module.datasets}

    @classmethod
    def seed(cls, seed: int):
        [module.seed(seed) for module in cls.__MODULES]

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls.__dset2module.keys()

    @classmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        return cls._get_module(dataset).get_tasks(dataset)

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        dm = cls._get_module(dataset)

        return dm.get_all_data(dataset, task)

    @classmethod
    def _get_module(cls, dataset) -> DataModule:
        try:
            return cls.__dset2module[dataset.upper()]
        except KeyError:
            raise InvalidDatasetError(dataset, cls.datasets)
