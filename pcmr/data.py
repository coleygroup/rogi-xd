from __future__ import annotations

from itertools import chain

import deepchem as dc
from deepchem import molnet
from deepchem.data import DiskDataset
import numpy as np

from pcmr.utils import Dataset


def combine_splits(
    splits: tuple[DiskDataset, DiskDataset, DiskDataset]
) -> tuple[list[str], np.ndarray]:
    smiss, ys = zip(*[(split.ids.tolist(), split.y) for split in splits])
    smis = list(chain(*smiss))
    y = np.concatenate(ys)

    return smis, y


def get_data(dataset: Dataset) -> tuple[list[str], np.ndarray, list[str]]:
    if dataset == Dataset.BACE:
        task_names, splits, _ = molnet.load_bace_regression()
    elif dataset == Dataset.CLEARANCE:
        task_names, splits, _ = molnet.load_clearance()
    elif dataset == Dataset.CLINTOX:
        task_names, splits, _ = molnet.load_clintox()
    elif dataset == Dataset.DELANEY:
        task_names, splits, _ = molnet.load_delaney()
    elif dataset == Dataset.FREESOLV:
        task_names, splits, _ = molnet.load_freesolv()
    elif dataset == Dataset.LIPO:
        task_names, splits, _ = molnet.load_lipo()
    elif dataset == Dataset.PDBBIND:
        task_names, splits, _ = molnet.load_pdbbind(set="refined")
    else:
        raise TypeError
    
    smis, y = combine_splits(splits)

    return smis, y, task_names