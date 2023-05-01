from typing import Optional

import pandas as pd

from rogi_xd.data.base import DataModule
from rogi_xd.data.tdc import TdcDataModule
from rogi_xd.data.guacamol import GuacaMolDataModule
from rogi_xd.data.dockstring import DockstringDataModule
from rogi_xd.utils.exceptions import InvalidDatasetError


class CompositeDataModule(DataModule):
    __MODULES: list[DataModule] = [TdcDataModule, GuacaMolDataModule, DockstringDataModule]
    __dset2module = {dset: module for module in __MODULES for dset in module.datasets}

    @classmethod
    def seed(cls, seed: int):
        [module.seed(seed) for module in cls.__MODULES]

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls.__dset2module.keys()

    @classmethod
    def tasks(cls, dataset: str) -> set[str]:
        return cls._get_module(dataset).tasks(dataset)

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
