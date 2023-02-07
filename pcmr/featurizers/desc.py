from typing import Iterable, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry

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


@FeaturizerRegistry.register("descriptor")
class DescriptorFeauturizer(FeaturizerBase):
    def __init__(self, names: Optional[Iterable[str]] = None, scale: bool = True, **kwargs):
        self.names = set(names or DEFAULT_DESCRIPTORS)
        self.scale = scale

        self.__funcs = [func for name, func in Descriptors.descList if name in self.names]

        super().__init__(**kwargs)

    def __call__(self, smis):
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        xss = [[func(mol) for func in self.__funcs] for mol in tqdm(mols)]
        X = np.array(xss)

        return MinMaxScaler().fit_transform(X) if self.scale else X

    def __str__(self) -> str:
        return "descriptor"
