from typing import Iterable, Union

import numpy as np
import selfies as sf
import torch
from transformers import pipeline

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.utils import select_device


@FeaturizerRegistry.register(alias="chemgpt")
class ChemGPTFeaturizer(FeaturizerBase):
    CHEMGPT = "ncfrey/ChemGPT-1.2B"

    def __init__(
        self, batch_size: int = None, device: Union[int, str, torch.device, None] = None
    ) -> None:
        device = select_device(device)
        self.fe = pipeline(
            "feature-extraction",
            model=self.CHEMGPT,
            device=device,
            framework="pt",
            return_tensors=True,
        )
        self.fe.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.batch_size = batch_size

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        sfs = [sf.encoder(smi) for smi in smis]
        return torch.stack([H[0, -1, :] for H in self.fe(sfs)]).numpy()
