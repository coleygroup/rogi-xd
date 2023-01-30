from abc import abstractmethod
import logging
from typing import Iterable, Optional, Union

import numpy as np
import torch
from transformers import pipeline

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.utils import select_device

logger = logging.getLogger(__name__)


class HuggingFaceFeaturizer(FeaturizerBase):
    @classmethod
    @abstractmethod
    def MODEL_ID(self) -> int:
        """the model identifier in the huggingface model hub"""

    @classmethod
    @abstractmethod
    def DEFAULT_BATCH_SIZE(self) -> int:
        """the default batch size"""

    @classmethod
    @abstractmethod
    def CLASSIFICATION_TOKEN_IDX(self) -> int:
        """the index of the token from which to calculate word embeddings"""

    def __init__(
        self, batch_size: Optional[int] = None, device: Union[int, str, torch.device, None] = None
    ) -> None:
        self.batch_size = batch_size
        self.fe = pipeline(
            "feature-extraction",
            model=self.MODEL_ID,
            device=select_device(device),
            framework="pt",
            return_tensors=True,
        )
        self.fe.tokenizer.padding_size = "right"

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: Optional[int]):
        if batch_size is None:
            logger.info(
                f"'batch_size' was `None`. Using default batch size (={self.DEFAULT_BATCH_SIZE})"
            )
            batch_size = self.DEFAULT_BATCH_SIZE

        if batch_size < 1:
            raise ValueError(f"'batch_size' cannot be < 1! got: {batch_size}")

        self.__batch_size = batch_size

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        return torch.stack([H[0, self.CLASSIFICATION_TOKEN_IDX, :] for H in self.fe(smis)]).numpy()


@FeaturizerRegistry.register(alias="chemberta")
class ChemBERTaFeaturizer(HuggingFaceFeaturizer):
    MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
    DEFAULT_BATCH_SIZE = 32
    CLASSIFICATION_TOKEN_IDX = 0


@FeaturizerRegistry.register(alias="chemgpt")
class ChemGPTFeaturizer(HuggingFaceFeaturizer):
    CHEMGPT = "ncfrey/ChemGPT-1.2B"
    DEFAULT_BATCH_SIZE = 8
    CLASSIFICATION_TOKEN_IDX = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fe.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.fe.tokenizer.padding_size = "left"
