from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

from torch import nn

from pcmr.models.vae.modules import RnnDecoder, RnnEncoder
from pcmr.models.vae.regularizers import Regularizer
from pcmr.models.vae.samplers import Sampler
from pcmr.models.vae.tokenizer import Tokenizer
from pcmr.utils.utils import Configurable


@dataclass
class FactoryConfig:
    alias: str
    config: Config


@dataclass
class TokenizerConfig:
    pattern: str
    tokens: Iterable[str]
    st: dict[str, str]


@dataclass
class EmbeddingConfig:
    num_embeddings: int
    embedding_dim: int
    padding_idx: int


@dataclass
class RnnEncoderConfig:
    embedding: nn.Embedding
    d_h: int
    n_layers: int
    dropout: float
    bidir: bool
    d_z: int
    regularizer: Regularizer


@dataclass
class RnnDecoderConfig:
    SOS: int
    EOS: int
    embedding: nn.Embedding
    d_z: int
    d_h: int
    n_layers: int
    dropout: float
    sampler: Sampler


@dataclass
class VAEConfig(Config):
    tokenizer: Tokenizer
    encoder: RnnEncoder
    decoder: RnnDecoder
    lr: float
    v_reg: FactoryConfig
    shared_emb: bool


def to_config(self) -> dict:
    return {
        "tokenizer": self.tokenizer.to_config(),
        "encoder": self.encoder.to_config(),
        "decoder": self.decoder.to_config(),
        "lr": self.lr,
        "v_reg": {"alias": self.v_reg.alias, "config": self.v_reg.to_config()},
        "shared_enb": self.encoder.emb is self.decoder.emb,
    }