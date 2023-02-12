from abc import abstractmethod
import logging
from typing import Iterable, Optional, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
import selfies as sf
import torch
from transformers import pipeline

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.featurizers.mixins import BatchSizeMixin
from pcmr.utils.utils import select_device

logger = logging.getLogger(__name__)


class HuggingFaceFeaturizerMixin(BatchSizeMixin):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        device: Union[int, str, torch.device, None] = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.fe = pipeline(
            "feature-extraction",
            model=self.MODEL_ID,
            device=select_device(device),
            framework="pt",
            return_tensors=True,
            **dict(truncation=True),
        )
        self.fe.tokenizer.padding_size = "right"

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        return torch.stack([H[0, self.CLASS_TOKEN_IDX, :] for H in self.fe(smis)]).numpy()

    def finetune(self, smis: Iterable[str], targets: ArrayLike) -> Self:
        raise NotImplementedError


@FeaturizerRegistry.register("chemberta")
class ChemBERTaFeaturizer(HuggingFaceFeaturizerMixin, FeaturizerBase):
    MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
    DEFAULT_BATCH_SIZE = 32
    CLASS_TOKEN_IDX = 0

    def __str__(self) -> str:
        return "chemberta"


@FeaturizerRegistry.register("chemgpt")
class ChemGPTFeaturizer(HuggingFaceFeaturizerMixin, FeaturizerBase):
    MODEL_ID = "ncfrey/ChemGPT-1.2B"
    DEFAULT_BATCH_SIZE = 1
    CLASS_TOKEN_IDX = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fe.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.fe.tokenizer.padding_size = "left"

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        return super().__call__([sf.encoder(smi, False) for smi in smis])

    def __str__(self) -> str:
        return "chemgpt"
