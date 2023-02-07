from __future__ import annotations

from abc import abstractmethod

from torch import Tensor, nn
from torch.distributions import Distribution, Categorical

from pcmr.utils import ClassRegistry, Configurable

SamplerRegistry = ClassRegistry()


class Sampler(nn.Module, Configurable):
    """A `Sampler` defines the sampling operation from a collection of unnormalized probabilities
    ("logits")"""

    def forward(self, logits: Tensor) -> Tensor:
        """Sample an index from last dimension of the input tensor

        Parameters
        ----------
        logits : Tensor
            a tensor of shape `... x d` containing unnormalized probabilities

        Returns
        --------
        Tensor
            a tensor of shape `...` containing the index of the selected item from the last dimension
        """
        return self.sample(logits.softmax(-1))

    @abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        """This does the same as :meth:`~.forward` but for normalized probablities.

        Depending on the subclass implementation, this may raise an error if the probabilities are not normalized
        """

    def to_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict) -> Sampler:
        return cls(**config)

@SamplerRegistry.register("mode")
class ModeSampler(Sampler):
    """A `ModeSampler` selects the index of the mode of the distribution"""

    def sample(self, probs: Tensor) -> Tensor:
        return probs.argmax(-1)


@SamplerRegistry.register("multinomial")
class MultinomialSampler(Sampler):
    """A `MultinomialSampler` selects an index by sampling from a multinomial ("categorical")
    distribution defined by the input probabilities"""

    def sample(self, probs: Tensor) -> Tensor:
        return Categorical(probs).sample()


@SamplerRegistry.register("noisy")
class NoisySampler(Sampler):
    """A `NoisySampler` adds noise sampled from the input distribution to the calculated
    probabilities before sampling based on the input :class:`~samplers.Sampler`"""

    def __init__(self, sampler: Sampler, noise: Distribution):
        self.sampler = sampler
        self.noise = noise

    def forward(self, logits: Tensor) -> Tensor:
        return self.sample(logits.softmax(-1))

    def sample(self, probs: Tensor) -> Tensor:
        return self.sampler.sample(probs + self.noise.sample(probs.shape))

    def to_config(self) -> dict:
        raise NotImplementedError(f"{self.alias} can not be serialized!")
    
    @classmethod
    def from_config(cls, config: dict) -> Sampler:
        sampler = config["sampler"]
        noise = config["noise"]

        return cls(sampler, noise)