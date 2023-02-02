from typing import Sequence, Union
import warnings

import pytorch_lightning as pl
from rdkit import Chem
import torch
from torch import Tensor, optim, nn
from torch.nn.utils import rnn

from pcmr.models.vae.tokenizer import Tokenizer
from pcmr.models.vae.modules import CharEncoder, CharDecoder
from pcmr.models.vae.schedulers import LinearScheduler, Scheduler, DummyScheduler
from pcmr.models.vae.utils import reconstruction_accuracy

warnings.filterwarnings("ignore", "Trying to infer the `batch_size`", UserWarning)
warnings.filterwarnings("ignore", "dropout option adds dropout after all but last", UserWarning)


class TrieAutoencoder(pl.LightningModule):
    """A character autoencoder for learning latent representations of strings

    Parameters
    ----------
    tokenizer : Tokenizer
        the :class:`Tokenizer` to use when measuring quality during validation
    encoder: CharEncoder
        the encoder module to project from tokenized sequences into the latent space
    decoder: TrieDecoder
        the decoder module to generate tokenized sequences from latent representations
    supervisor: Supervisor
        the supervision module to use for latent space organization with labelled inputs
    lr : float, default=3e-4
        the learning rate
    v_reg : Union[float, Scheduler, None], default=None
        the regularization loss weight scheduler. If `None`, use a linear scheduler from 0->0.1 over
        100 epochs. If a float value is supplied, use a constant weight
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

        if encoder.d_z != decoder.d_z:
            raise ValueError(
                "encoder and decoder have mismatched latent dimension sizes! "
                f"got: {encoder.d_z} and {decoder.d_z}, respectively."
            )

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder

        self.lr = lr
        if v_reg is None:
            self.v_reg = LinearScheduler(0, 0.1, 100)
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
        return self.decoder(self.encoder(xs))

    def on_train_start(self):
        self.v_reg.i = 0

    def training_step(self, batch: tuple, batch_idx) -> Tensor:
        xs, Y, mask_idxs = batch

        Z, l_reg = self.encoder.forward_step(xs)
        X_logits = self.decoder.forward_step(xs, Z, mask_idxs)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.tokenizer.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)
        l_sup = self.supervisor(Z, Y)

        self.log("train/rec", l_rec)
        self.log("train/reg", l_reg)
        self.log("train/sup", l_sup)
        self.log("loss", l_rec + l_reg + l_sup)

        return l_rec + self.v_reg.v * l_reg + self.v_sup * l_sup

    def validation_step(self, batch, batch_idx):
        xs, Y, mask_idxsss = batch

        Z, l_reg = self.encoder.forward_step(xs)
        X_logits = self.decoder.forward_step(xs, Z, mask_idxsss)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.tokenizer.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)
        rec_acc = reconstruction_accuracy(xs, self.reconstruct(xs))

        return l_rec, l_reg, rec_acc

    def predict_step(self, batch, batch_idx: int, dataloader_idx=0) -> Tensor:
        return self.encode(batch)

    def training_epoch_end(self, *args):
        self.log(f"v/{self.encoder.regularizer.name}", self.v_reg.v)
        self.v_reg.step()

    def validation_epoch_end(self, outputs):
        l_recs, l_regs, accs = zip(*outputs)
        n = 1000

        L = torch.tensor(list(zip(l_recs, l_regs))).mean(0)
        accuracy = sum(accs) / len(accs)
        f_valid, f_unique = self.measure_quality(torch.randn(n, self.d_z, device=self.device))

        self.log("val/rec", L[0])
        self.log("val/reg", L[1])
        self.log("val/loss", L.sum())
        self.log("val/accuracy", accuracy)
        self.log(f"val/valid@{n//1000}k", f_valid)
        self.log(f"val/unique@{n//1000}k", f_unique)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()},
        ]

        return optim.Adam(param_groups, self.lr)

    def measure_quality(self, Z: Tensor):
        smis = ["".join(self.tokenizer.ids2tokens(x.tolist())) for x in self.generate(Z)]

        valid_smis = [smi for smi in smis if Chem.MolFromSmiles(smi) is not None]
        f_valid = len(valid_smis) / len(Z)
        try:
            f_unique = len(set(valid_smis)) / len(valid_smis)
        except ZeroDivisionError:
            f_unique = 0

        return f_valid, f_unique
