import numpy as np
import pytest
from rdkit import Chem

from pcmr.rogi import RoughnessIndex
from pcmr.utils import Metric


@pytest.fixture
def smis():
    return ["c1ccccc1", "CCCC", "CC(=O)C"]


@pytest.fixture
def mols(smis):
    return [Chem.MolFromSmiles(smi) for smi in smis]


@pytest.fixture
def fps(mols):
    return [Chem.RDKFingerprint(m) for m in mols]


def test_dist_mat(fps):
    D1 = RoughnessIndex.compute_distance_matrix(fps)
    D2 = RoughnessIndex.compute_distance_matrix2(fps)

    np.testing.assert_array_almost_equal(D1, D2)