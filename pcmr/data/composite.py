from typing import Optional

import pandas as pd

from pcmr.data.base import DataModule
from pcmr.data.tdc import TdcDataModule
from pcmr.data.guacamol import GuacaMolDataModule
from pcmr.data.molnet import MoleculeNetDataModule


class CompositeDataModule(DataModule):
    __MODULES: list[DataModule] = [TdcDataModule, GuacaMolDataModule, MoleculeNetDataModule]
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
        cls.check_dataset(dataset)

        return cls.__dset2module[dataset.upper()].get_tasks(dataset)

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        cls.check_dataset(dataset)

        dm = cls.__dset2module[dataset.upper()]

        if task is not None and task not in dm.get_tasks(dataset):
            raise ValueError(f"Invalid task! '{task}' is not a valid task for '{dataset}' dataset.")

        return dm.get_all_data(dataset, task)