from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataModule(ABC):
    @classmethod
    def get_data(cls, dataset: str, task: Optional[str] = None, n: int = 10000) -> pd.DataFrame:
        df = cls.get_all_data(dataset, task)
        
        return df.sample(n) if len(df) > n else df

    @classmethod
    @property
    @abstractmethod
    def datasets(cls) -> set[str]:
        return cls._DATASETS

    @classmethod
    @abstractmethod
    def get_tasks(cls, dataset: str) -> set[str]:
        pass

    @classmethod
    @abstractmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        pass

    