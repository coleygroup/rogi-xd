# from typing import Iterable

# import numpy as np
# from rdkit import Chem

# from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry


# @FeaturizerRegistry.register(alias="morgan")
# class MorganFeauturizer(FeaturizerBase):
#     def __init__(self, radius: int = 2, length: int = 2048):
#         self.radius = radius
#         self.length = length

#     def __call__(self, smis: Iterable[str]) -> np.ndarray:
#         mols = [Chem.MolFromSmiles(smi) for smi in smis]
#         return [
#             Chem.GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.length)
#             for m in mols
#         ]
