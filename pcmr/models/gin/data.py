from torchdrug.core import Registry as R
from torchdrug.data.dataset import MoleculeDataset

@R.register("datasets.Custom")
class CustomDataset(MoleculeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
