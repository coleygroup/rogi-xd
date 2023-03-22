import logging
from typing import Iterable, Optional, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
import selfies as sf
import torch
from transformers import pipeline, AutoModel, AutoConfig

from pcmr.featurizers.base import FeaturizerBase, FeaturizerRegistry
from pcmr.featurizers.mixins import BatchSizeMixin
from pcmr.utils.utils import select_device

logger = logging.getLogger(__name__)


class HuggingFaceFeaturizerMixin(BatchSizeMixin):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        device: Union[int, str, torch.device, None] = None,
        reinit: bool = False,
        **kwargs,
    ):
        self.batch_size = batch_size
        if reinit:
            model = AutoModel.from_config(AutoConfig.from_pretrained(self.MODEL_ID))
        else:
            model = AutoModel.from_pretrained(self.MODEL_ID)

        self.fe = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=self.MODEL_ID,
            device=select_device(device),
            framework="pt",
            return_tensors=True,
            truncation=True,
        )
        self.fe.tokenizer.padding_size = "right"

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        Hs = [H[0, self.CLASS_TOKEN_IDX, :] for H in self.fe(smis, batch_size=self.batch_size)]

        return torch.stack(Hs).numpy().astype(float)

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        raise NotImplementedError


@FeaturizerRegistry.register("chemberta")
class ChemBERTaFeaturizer(HuggingFaceFeaturizerMixin, FeaturizerBase):
    MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
    DEFAULT_BATCH_SIZE = 32
    CLASS_TOKEN_IDX = 0


@FeaturizerRegistry.register("chemgpt")
class ChemGPTFeaturizer(HuggingFaceFeaturizerMixin, FeaturizerBase):
    MODEL_ID = "ncfrey/ChemGPT-1.2B"
    DEFAULT_BATCH_SIZE = 8
    CLASS_TOKEN_IDX = -1

    def __init__(
        self,
        batch_size: Optional[int] = None,
        device: Union[int, str, torch.device, None] = None,
        reinit: bool = False,
        **kwargs,
    ):
        super().__init__(batch_size, device, reinit, **kwargs)

        self.fe.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.fe.tokenizer.padding_side = "left"

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        return super().__call__([sf.encoder(smi, False) for smi in smis])
