from typing import Iterable

from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.utils.rogi import Metric


@FeaturizerRegistry.register(alias="morgan")
class MorganFeauturizer(FeaturizerBase):
    metric = Metric.TANIMOTO

    def __init__(self, radius: int = 2, length: int = 512, **kwargs):
        self.radius = radius
        self.length = length

    def __call__(self, smis: Iterable[str]) -> list[ExplicitBitVect]:
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        return [
            Chem.GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.length)
            for m in mols
        ]
