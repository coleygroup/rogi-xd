from typing import Optional, Sequence
import warnings

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from pcmr.models.vae.regularizers import Regularizer, VariationalRegularizer
from pcmr.models.vae.tokenizer import Tokenizer
from pcmr.models.vae.samplers import ModeSampler, Sampler


class CharEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        d_emb: int = 64,
        d_h: int = 256,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidir: bool = True,
        d_z: int = 128,
        regularizer: Optional[Regularizer] = None,
    ):
        super().__init__()

        self.d_z = d_z

        self.emb = embedding
        self.rnn = nn.GRU(
            d_emb, d_h, n_layers, batch_first=True, dropout=dropout, bidirectional=bidir
        )
        d_h_rnn = 2 * d_h if bidir else d_h
        
        self.regularizer = regularizer or VariationalRegularizer(d_z)
        self.regularizer.setup(d_h_rnn)

    @property
    def d_z(self) -> int:
        return self.regularizer.d_z

    def _forward(self, xs: Sequence[Tensor]) -> Tensor:
        xs_emb = [self.emb(x) for x in xs]
        X = rnn.pack_sequence(xs_emb, enforce_sorted=False)

        _, H = self.rnn(X)
        H = H[-(1 + int(self.rnn.bidirectional)) :]

        return torch.cat(H.split(1), -1).squeeze(0)

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        return self.regularizer(self._forward(xs))

    def forward_step(self, xs: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        return self.regularizer.forward_step(self._forward(xs))


class CharDecoder(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding: nn.Embedding,
        d_z: int = 128,
        d_h: int = 512,
        n_layers: int = 3,
        dropout: float = 0.2,
        sampler: Optional[Sampler] = None,
    ):
        super().__init__()

        if len(tokenizer) != embedding.num_embeddings:
            warnings.warn(
                "Input 'tokenizer' and 'embedding' have mismatching vocabulary sizes!"
                f"got: {len(tokenizer)} and {embedding.num_embeddings}, respectively."
            )

        self.SOS = tokenizer.SOS
        self.EOS = tokenizer.EOS
        self.PAD = tokenizer.PAD

        self.emb = embedding
        self.d_z = d_z
        self.d_v = embedding.num_embeddings
        
        self.z2h = nn.Linear(self.d_z, d_h)
        self.rnn = nn.GRU(self.emb.embedding_dim, d_h, n_layers, batch_first=True, dropout=dropout)
        self.h2v = nn.Linear(d_h, self.d_v)
        self.sampler = sampler or ModeSampler()

    def forward_step(self, xs: Sequence[Tensor], Z: Tensor) -> Tensor:
        lengths = [len(x) for x in xs]
        X = rnn.pad_sequence(xs, batch_first=True, padding_value=self.PAD)

        X_emb = self.emb(X)
        X_packed = rnn.pack_padded_sequence(X_emb, lengths, batch_first=True, enforce_sorted=False)
        H = self.z2h(Z)
        H_0 = H.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        O_packed, _ = self.rnn(X_packed, H_0)
        O, _ = nn.utils.rnn.pad_packed_sequence(O_packed, True)

        return self.h2v(O)

    def forward(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
        n = len(Z)

        x_t = torch.tensor(self.SOS, device=Z.device).repeat(n)
        X_gen = torch.tensor([self.PAD], device=Z.device).repeat(n, max_len)
        X_gen[:, 0] = self.SOS

        seq_lens = torch.tensor([max_len], device=Z.device).repeat(n)
        eos_mask = torch.zeros(n, dtype=torch.bool, device=Z.device)

        H_t = self.z2h(Z).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        for t in range(1, max_len):
            x_emb = self.emb(x_t).unsqueeze(1)
            O, H_t = self.rnn(x_emb, H_t)
            logits = self.h2v(O.squeeze(1)).softmax(-1)

            x_t = self.sampler(logits)
            X_gen[~eos_mask, t] = x_t[~eos_mask]

            eos_mask_t = ~eos_mask & (x_t == self.EOS)
            seq_lens[eos_mask_t] = t + 1
            eos_mask = eos_mask | eos_mask_t

        return [X_gen[i, : seq_lens[i]] for i in range(len(X_gen))]
