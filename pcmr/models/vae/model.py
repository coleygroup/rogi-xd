from typing import Sequence, Union
import warnings

import pytorch_lightning as pl
from rdkit import Chem
from rdkit.rdBase import BlockLogs
import torch
from torch import Tensor, optim, nn
from torch.nn.utils import rnn

from pcmr.models.utils import PlMixin
from pcmr.models.vae.tokenizer import Tokenizer
from pcmr.models.vae.modules import CharEncoder, CharDecoder
from pcmr.models.vae.schedulers import LinearScheduler, Scheduler, DummyScheduler

block = BlockLogs()
warnings.filterwarnings("ignore", "Trying to infer the `batch_size`", UserWarning)
warnings.filterwarnings("ignore", "dropout option adds dropout after all but last", UserWarning)


class LitVAE(pl.LightningModule, PlMixin):
    """A character autoencoder for learning latent representations of strings

    Parameters
    ----------
    tokenizer : Tokenizer
        the :class:`~pcmr.models.vae.tokenizer.Tokenizer` to use when measuring quality during
        validation
    encoder: CharEncoder
        the encoder module to project from tokenized sequences into the latent space
    decoder: CharDecoder
        the decoder module to generate tokenized sequences from latent representations
    lr : float, default=3e-4
        the learning rate
    v_reg : Union[float, Scheduler, None], default=None
        the regularization loss weight scheduler. If `None`, use a linear scheduler from 0->0.1 over
        20 epochs. If a float value is supplied, use a constant weight
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        encoder: CharEncoder,
        decoder: CharDecoder,
        lr: float = 3e-4,
        v_reg: Union[float, Scheduler, None] = None,
    ):
        super().__init__()

        if len(tokenizer) != encoder.emb.num_embeddings:
            raise ValueError(
                "tokenizer and encoder have mismatched vocabulary sizes! "
                f"got: {len(tokenizer)} and {encoder.emb.num_embeddings}, respectively."
            )
        if encoder.d_z != decoder.d_z:
            raise ValueError(
                "'encoder' and 'decoder' have mismatched latent dimension sizes! "
                f"got: {encoder.d_z} and {decoder.d_z}, respectively."
            )
        if encoder.emb is not decoder.emb:
            warnings.warn(
                "encoder and decoder are using different embedding layers! Is this intentional?"
            )

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        if v_reg is None:
            self.v_reg = LinearScheduler(0, 0.1, 20)
        elif isinstance(v_reg, float):
            self.v_reg = DummyScheduler(v_reg)
        else:
            self.v_reg = v_reg

        self.rec_metric = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.tokenizer.PAD)

    @property
    def d_z(self) -> int:
        return self.encoder.d_z

    def encode(self, xs: Sequence[Tensor]) -> Tensor:
        return self.encoder(xs)

    def decode(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
        return self.decoder(Z, max_len)

    def reconstruct(self, xs: Sequence[Tensor]) -> list[Tensor]:
        return self.decode(self.encode(xs))

    forward = encode
    generate = decode

    def on_train_start(self):
        self.v_reg.i = 0

    def training_step(self, batch: Sequence[Tensor], batch_idx) -> Tensor:
        xs = batch

        Z, l_reg = self.encoder.forward_step(xs)
        X_logits = self.decoder.forward_step(xs, Z)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.tokenizer.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)

        metrics = dict(rec=l_rec, reg=l_reg)
        self._log_split("train", metrics)
        self.log("loss", l_rec + l_reg)

        return l_rec + self.v_reg.v * l_reg

    def validation_step(self, batch: Sequence[Tensor], batch_idx):
        xs = batch

        Z, l_reg = self.encoder.forward_step(xs)
        X_logits = self.decoder.forward_step(xs, Z)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.tokenizer.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)
        acc = sum(map(torch.equal, xs, self.reconstruct(xs)))

        return l_rec, l_reg, acc, len(xs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=0) -> Tensor:
        return self.encode(batch)

    def on_train_epoch_start(self):
        self.log(f"v/{self.encoder.reg.alias}", self.v_reg.v)

    def training_epoch_end(self, *args):
        self.v_reg.step()

    def validation_epoch_end(self, outputs):
        l_recs, l_regs, accs, sizes = torch.tensor(outputs).hsplit(4)
        l_rec, l_reg, acc = ((l * sizes).sum() / sizes.sum() for l in [l_recs, l_regs, accs])
        metrics = dict(rec=l_rec, reg=l_reg, loss=l_rec + l_reg, acc=acc)

        n = 1000
        f_valid, f_unique = self.check_gen_quality(torch.randn(n, self.d_z, device=self.device))
        metrics.update({f"valid@{n//1000}k": f_valid, f"unique@{n//1000}k": f_unique})

        self._log_split("val", metrics)

    def configure_optimizers(self):
        params = set(self.encoder.parameters()) | set(self.decoder.parameters())

        return optim.Adam(params, self.lr)

    def check_gen_quality(self, Z: Tensor):
        smis = [self.tokenizer.decode(x.tolist()) for x in self.decode(Z)]
        smis = [smi for smi in smis if Chem.MolFromSmiles(smi) is not None]

        f_valid = len(smis) / len(Z)
        f_unique = 0 if len(smis) == 0 else len(set(smis)) / len(smis)

        return f_valid, f_unique
