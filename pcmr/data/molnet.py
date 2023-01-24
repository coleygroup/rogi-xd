from __future__ import annotations

from itertools import chain
from typing import Optional

from deepchem import molnet
from deepchem.data import DiskDataset
import numpy as np
import pandas as pd

from pcmr.data.base import DataModule


def combine_splits(splits: tuple[DiskDataset, ...]) -> tuple[list[str], np.ndarray]:
    smiss, ys = zip(*[(split.ids.tolist(), split.y) for split in splits])
    smis = list(chain(*smiss))
    y = np.concatenate(ys)

    return smis, y


class MoleculeNetDataModule(DataModule):
    __DATASETS = {"BACE", "CLEARANCE", "CLINTOX", "DELANEY", "FREESOLV", "LIPO", "PDBBIND"}

    @classmethod
    def get_full_dataset(cls, dataset) -> tuple[list[str], tuple[DiskDataset, ...]]:
        cls.check_dataset(dataset)
        
        dataset_ = dataset.upper()
        if dataset_ == "BACE":
            task_names, splits, _ = molnet.load_bace_regression()
        elif dataset_ == "CLEARANCE":
            task_names, splits, _ = molnet.load_clearance()
        elif dataset_ == "CLINTOX":
            task_names, splits, _ = molnet.load_clintox()
        elif dataset_ == "DELANEY":
            task_names, splits, _ = molnet.load_delaney()
        elif dataset_ == "FREESOLV":
            task_names, splits, _ = molnet.load_freesolv()
        elif dataset_ == "LIPO":
            task_names, splits, _ = molnet.load_lipo()
        elif dataset_ == "PDBBIND":
            task_names, splits, _ = molnet.load_pdbbind(set="refined")
        else:
            raise ValueError(f"Invalid MoleculeNet dataset! got: {dataset}")

        return task_names, splits

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return cls.__DATASETS

    @classmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        cls.check_dataset(dataset)
        tasks, _ = cls.get_full_dataset(dataset)

        return tasks

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:        
        tasks, splits = cls.get_full_dataset(dataset)
        smis, Y = combine_splits(splits)
        try:
            j = 0 if task is None else tasks.index(task)
        except ValueError:
            raise ValueError(f"Invalid task! got: '{task}' but expected one of {tasks} or `None`!")
        y = Y[:, j]

        return pd.DataFrame(dict(smiles=smis, y=y))
