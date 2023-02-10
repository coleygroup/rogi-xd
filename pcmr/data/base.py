from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Iterable, Optional

import pandas as pd

from pcmr.utils.exceptions import InvalidDatasetError, InvalidTaskError

logger = logging.getLogger(__name__)


class DataModule(ABC):
    @classmethod
    def seed(cls, seed: int):
        logger.info(f"{cls.__name__} is not randomly generated! Nothing happened...")

    @classmethod
    def get_data(
        cls, dataset: str, task: Optional[str] = None, n: int = 10000, seed: Optional[int] = None
    ) -> pd.DataFrame:
        """get `n` rows of data from the associated (dataset, task) pair

        Parameters
        ----------
        dataset : str
            the parent dataset to get
        task : Optional[str], default=None
            the specific task to retrieve data for. If None,
        n : int, default=10000
            the maximum amount of datapoints to return. If `n` is larger than the full dataset,
            return the full dataset. Otherwise, randomly subsample the dataset
        seed : Optional[int], default=None
            the random seed to use when subsampling the data (if necessary).

        Returns
        -------
        pd.DataFrame
            a `DataFrame` containing two columns: `smiles` and `y` that hold the SMILES strings and
            property values, respectively, of the data.
        """
        df = cls.get_all_data(dataset, task)

        if len(df) > n:
            logger.info(f"Subsampling {n} rows from dataframe with {len(df)} rows")
            return df.sample(n, random_state=seed)

        return df

    @classmethod
    @property
    @abstractmethod
    def datasets(cls) -> set[str]:
        """The possible datasets available from this DataModule"""

    @classmethod
    @abstractmethod
    def get_tasks(cls, dataset: str) -> Iterable[str]:
        """The tasks associated with the given dataset, if any"""

    @classmethod
    @abstractmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        """get all data from the associated (dataset, task) pair.

        Parameters
        ----------
        dataset : str
            the dataset to retrieve data for
        task : Optional[str], default=None
            the associated task name. This should only be specified in the case of datasets with
            multiple tasks.

        Returns
        -------
        pd.DataFrame
            a `DataFrame` containing two columns: `smiles` and `y` that hold the SMILES strings and
            property values, respectively, of the data.

        Raises
        ------
        ValueError
            if the input `task` is not associated the given `dataset`
        """

    @classmethod
    def check_dataset(cls, dataset: str):
        if dataset.upper() not in cls.datasets:
            raise InvalidDatasetError(dataset, cls.datasets)

    @classmethod
    def check_task(cls, dataset: str, task: Optional[str]):
        tasks = cls.get_tasks(dataset)
        if task in tasks or (task is None and len(tasks) < 2):
            return

        raise InvalidTaskError(task, dataset, tasks)
