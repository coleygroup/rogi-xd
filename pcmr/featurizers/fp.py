from typing import Iterable, Optional
from typing_extensions import Self

from numpy.typing import ArrayLike
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import ExplicitBitVect
from tqdm import tqdm

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.utils.rogi import Metric


@FeaturizerRegistry.register(alias="morgan")
class MorganFeauturizer(FeaturizerBase):
    metric = Metric.TANIMOTO

    def __init__(self, radius: int = 2, length: Optional[int] = 512, **kwargs):
        self.radius = radius
        self.length = length or 512
        self.quiet = False

    def __call__(self, smis: Iterable[str]) -> list[ExplicitBitVect]:
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        return [
            Chem.GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.length)
            for m in tqdm(mols, "Featurizing", leave=False, disable=self.quiet)
        ]

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        return self
