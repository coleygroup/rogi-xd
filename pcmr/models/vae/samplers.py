from abc import abstractmethod

from torch import Tensor, nn
from torch.distributions import Distribution, Categorical


class Sampler(nn.Module):
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


class ModeSampler(Sampler):
    """A `ModeSampler` selects the index of the mode of the distribution"""

    def sample(self, probs: Tensor) -> Tensor:
        return probs.argmax(1)


class MultinomialSampler(Sampler):
    """A `MultinomialSampler` selects an index by sampling from a multinomial ("categorical")
    distribution defined by the input probabilities"""

    def sample(self, probs: Tensor) -> Tensor:
        return Categorical(probs).sample()


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
