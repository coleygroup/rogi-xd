from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn

from pcmr.utils import ClassRegistry, Configurable

RegularizerRegistry = ClassRegistry()


class Regularizer(nn.Module, Configurable):
    """A :class:`Regularizer` projects from the encoder output to the latent space and
    calculates the associated loss of that projection"""

    def __init__(self, d_z: int, **kwargs):
        super().__init__()
        self.d_z = d_z

    @property
    def name(self) -> str:
        """the name of the regularization loss"""

    @abstractmethod
    def setup(self, d_h: int):
        """Perform any setup necessary before using this `Regularizer`. NOTE: this function _must_
        be called at some point in the `__init__()` function."""

    @abstractmethod
    def forward(self, H: Tensor) -> Tensor:
        """Project the output of the encoder into the latent space and regularize it"""

    @abstractmethod
    def forward_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate both the regularized latent representation (i.e., `forward()`) and associated
        loss of the encoder output"""

    def to_config(self) -> dict:
        return {"d_z": self.d_z}

    @classmethod
    def from_config(cls, config: dict) -> Regularizer:
        return cls(**config)


@RegularizerRegistry.register("dummy")
class DummyRegularizer(Regularizer):
    """A :class:`DummyRegularizer` calculates no regularization loss"""

    def __init__(self, d_z: int):
        super().__init__(d_z)

    @property
    def name(self) -> str:
        return "ae"

    def setup(self, d_h: int):
        self.q_h2z_mu = nn.Linear(d_h, self.d_z)

    def forward(self, H: Tensor) -> Tensor:
        return self.q_h2z_mu(H)

    def forward_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        return self(H), torch.tensor(0.0)


@RegularizerRegistry.register("vae")
class VariationalRegularizer(DummyRegularizer):
    """A :class:`VariationalRegularizer` uses the reparameterization trick to project into to the
    latent space and calculates the regularization loss as the KL divergence between the output and
    a multivariate unit normal distribution

    References
    ----------
    .. [1] Kingma, D.P.; and Welling, M.; arXiv:1312.6114v10 [stat.ML], 2014
    """

    def __init__(self, d_z: int):
        super().__init__(d_z)

    @property
    def name(self) -> str:
        return "kl"

    def setup(self, d_h: int):
        super().setup(d_h)
        self.q_h2z_logvar = nn.Linear(d_h, self.d_z)

    def forward(self, H: Tensor):
        Z_mean, Z_logvar = self.q_h2z_mu(H), self.q_h2z_logvar(H)

        return self.reparameterize(Z_mean, Z_logvar)

    def forward_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        Z_mean, Z_logvar = self.q_h2z_mu(H), self.q_h2z_logvar(H)

        Z = self.reparameterize(Z_mean, Z_logvar)
        l_kl = 0.5 * (Z_mean**2 + Z_logvar.exp() - 1 - Z_logvar).sum(1).mean()

        return Z, l_kl

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        sd = (logvar / 2).exp()
        eps = torch.randn_like(sd)

        return mean + eps * sd
