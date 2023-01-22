from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataModule(ABC):
    @classmethod
    def seed(cls, seed: int):
        logger.info(f"{cls.__name__} is not randomly generated! Nothing happened...")

    @classmethod
    def get_data(
        cls, dataset: str, task: Optional[str] = None, n: int = 10000, seed: Optional[int] = None
    ) -> pd.DataFrame:
        df = cls.get_all_data(dataset, task)
        
        if len(df) > n:
            logger.info(f"Subsampling {n} rows from dataframe with {len(df)} rows")
            return df.sample(n, random_state=seed)
            
        return df

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

    