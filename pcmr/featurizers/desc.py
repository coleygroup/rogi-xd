import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry


@FeaturizerRegistry.register(alias="descriptor")
class DescriptorFeauturizer(FeaturizerBase):
    __DESC_NAMES, __DESC_FUNCS = zip(*Descriptors.descList)

    def __call__(self, smis):
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        xss = [[func(mol) for func in self.__DESC_FUNCS] for mol in tqdm(mols)]

        return np.array(xss)
