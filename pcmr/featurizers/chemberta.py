from typing import Iterable, Union

import numpy as np
import torch
from transformers import pipeline

from pcmr.featurizers.base import Featurizer


class ChemGPTFeaturizer(Featurizer):
    CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"

    def __init__(self, batch_size: int = None, device: Union[int, str, None] = None) -> None:
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = pipeline(
            "feature-extraction",
            model=self.CHEMBERTA,
            device=device,
            framework="pt",
            return_tensors=True
        )
        self.batch_size = batch_size

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        return torch.stack([H[0, 0, :] for H in self.fe(smis)]).numpy()
