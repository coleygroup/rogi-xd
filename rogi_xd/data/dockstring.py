import logging
from typing import Optional

import pandas as pd

from rogi_xd.data.base import DataModule
from rogi_xd.utils.utils import CACHE_DIR, download_file

logger = logging.getLogger(__name__)

DOCKSTRING_URL = "https://figshare.com/ndownloader/files/35948138"
DOCKSTRING_PATH = CACHE_DIR / "dockstring.tsv"

if not DOCKSTRING_PATH.exists() or DOCKSTRING_PATH.stat().st_size == 0:
    logger.debug("DOCKSTRING TSV not found. Downloading...")
    download_file(DOCKSTRING_URL, DOCKSTRING_PATH, "Downloading dockstring TSV")
else:
    logger.debug("Found local copy of DOCKSTRING TSV")


class DockstringDataModule(DataModule):
    """The :class:`DockstringDataModule` loads the Dockstring dataset from [1]_

    References
    ----------
    .. [1] García-Ortegón M.; Simm G.N.C.; Tripp A.J.; Hernández-Lobato J.M.; Bender A.; Bacallado
    S. J. Chem. Inf. Model. 2022, 62 (15). doi: 10.1021/acs.jcim.1c01334.

    """

    __df = pd.read_csv(DOCKSTRING_PATH, sep="\t").drop(columns="inchikey").set_index("smiles")

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return {"DOCKSTRING"}

    @classmethod
    def tasks(cls, dataset: str) -> set[str]:
        cls.check_dataset(dataset)

        return set(cls.__df)

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        cls.check_task(dataset, task)

        y = cls.__df[task.upper()]

        return pd.DataFrame(dict(smiles=cls.__df.index.values, y=y))
