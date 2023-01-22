from __future__ import annotations

from typing import Optional

import pandas as pd

from pcmr.data.base import DataModule
from pcmr.data.tdc import TdcDataModule
from pcmr.data.guacamol import GuacaMolDataModule
from pcmr.data.molnet import MoleculeNetDataModule

dset2module: dict[str, DataModule] = {
    **{d: TdcDataModule for d in TdcDataModule.datasets},
    **{d: GuacaMolDataModule for d in GuacaMolDataModule.datasets},
    **{d: MoleculeNetDataModule for d in MoleculeNetDataModule.datasets},
}

def get_data(dataset: str, task: Optional[str] = None, n: int = 10000) -> pd.DataFrame:
    try:
        dm = dset2module[dataset.upper()]
    except KeyError:
        raise ValueError(f"Invalid dataset! got: '{dataset}'.")
    
    if task is not None and task not in dm.get_tasks(dataset):
        raise ValueError(f"Invalid task! '{task}' is not a valid task for '{dataset}' dataset.")
    
    return dm.get_data(dataset, task, n)