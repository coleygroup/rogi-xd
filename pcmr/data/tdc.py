from __future__ import annotations
from typing import Optional

import pandas as pd
from tdc.single_pred import ADME, Tox
from tdc.utils import retrieve_label_name_list

from pcmr.data.base import DataModule


class TdcDataModule(DataModule):
    _ADME_DATASETS = {'CLEARANCE_HEPATOCYTE_AZ', 'SOLUBILITY_AQSOLDB', 'CLEARANCE_MICROSOME_AZ', 'VDSS_LOMBARDO', 'CACO2_WANG', 'HALF_LIFE_OBACH', 'LIPOPHILICITY_ASTRAZENECA', 'PPBR_AZ', 'HYDRATIONFREEENERGY_FREESOLV'}
    _TOX_DATASETS = {'HERG_CENTRAL', 'LD50_ZHU'}

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return {*cls._ADME_DATASETS, *cls._TOX_DATASETS}

    @classmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        return set(retrieve_label_name_list(dataset))

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        dataset_ = dataset.upper()
        if dataset_ in cls._ADME_DATASETS:
            df: pd.DataFrame = ADME(dataset_, label_name=task).get_data("df")
        elif dataset_ in cls._TOX_DATASETS:
            df: pd.DataFrame = Tox(dataset_, label_name=task).get_data("df")

        return df.rename(columns={"Drug": "smiles", "Y": "y"})
    