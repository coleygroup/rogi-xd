import logging
from typing import Callable, Iterable, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTORS = [
    "MolWt",
    "FractionCSP3",
    "NumHAcceptors",
    "NumHDonors",
    "NOCount",
    "NHOHCount",
    "NumAliphaticRings",
    "NumAliphaticHeterocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumRotatableBonds",
    "TPSA",
    "qed",
    "MolLogP",
]
DESC_TO_FUNC: dict[str, Callable[[Chem.Mol], float]] = dict(Descriptors.descList)


@FeaturizerRegistry.register("descriptor")
class DescriptorFeauturizer(FeaturizerBase):
    def __init__(self, descs: Optional[Iterable[str]] = None, scale: bool = True, **kwargs):
        self.descs = set(descs or DEFAULT_DESCRIPTORS)
        self.scale = scale
        self.quiet = False

        super().__init__(**kwargs)

    @property
    def descs(self) -> list[str]:
        return self.__names

    @descs.setter
    def descs(self, descs: Iterable[str]):
        self.__names = []
        self.__funcs = []
        invalid_names = []
        for desc in descs:
            func = DESC_TO_FUNC.get(desc)
            if func is None:
                invalid_names.append(desc)
            else:
                self.__names.append(desc)
                self.__funcs.append(func)

        if len(invalid_names) > 0:
            logger.info(f"Ignored invalid names: {invalid_names}.")

    def __len__(self) -> int:
        return len(self.__funcs)

    def __call__(self, smis):
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        xss = [
            [func(mol) for func in self.__funcs]
            for mol in tqdm(mols, leave=False, disable=self.quiet)
        ]
        X = np.array(xss)

        return MinMaxScaler().fit_transform(X) if self.scale else X

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        """a :class:`DescriptorFeaturizer` can't be finetuned"""

        return self

    def __str__(self) -> str:
        return "descriptor"
