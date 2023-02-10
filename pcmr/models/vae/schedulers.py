from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike

from pcmr.utils import ClassRegistry, Configurable

SchedulerRegistry = ClassRegistry()


class Scheduler(ABC, Configurable):
    """A Scheduler anneals a weight term from `v0` -> `v1` over `max_steps` number of steps

    Parameters
    ----------
    v_min : float, default=0
        the minimum weight
    v_max : float, default=1
        the maximum weight
    max_steps : int, default=sys.maxsize
        the number of steps to take before reaching v1
    """

    def __init__(self, v_min: float = 0, v_max: float = 1, max_steps: int = 1000, name: str = "v"):
        self.v_min = v_min
        self.v_max = v_max
        self.max_steps = max_steps
        self.name = name

        self.i = 0
        self.schedule = self.calc_schedule(v_min, v_max, max_steps)

    def __len__(self) -> int:
        """The number of steps in this scheduler"""
        return self.max_steps

    @property
    def v(self) -> float:
        """The current weight"""
        return self.schedule[min(len(self), self.i)]

    def step(self) -> float:
        """Step the scheduler and return the new weight"""
        self.i += 1
        return self.v

    @staticmethod
    @abstractmethod
    def calc_schedule(v_min, v_max, max_steps) -> np.ndarray:
        """Calculate the schedule of the KL weight according to this scheduler's underlying algorithm

        Parameters
        ----------
        v_min : float
            the minimum weight
        v_max : float
            the maximum weight
        max_steps : int
            the number of steps to take between `v0` and `v1`

        Returns
        -------
        np.ndarray
            a vector of length `max_steps` or `max_steps + 1` containing the KL weight at the given
            number of steps
        """

    def to_config(self) -> dict:
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "max_steps": self.max_steps,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        return cls(**config)

    def get_params(self) -> Iterable[tuple[str, Any]]:
        return self.to_config().items()


@SchedulerRegistry.register("constant")
class ConstantScheduler(Scheduler):
    """A dummy class to return a constant weight of v"""

    def __init__(self, v: float, name: str = "v"):
        super().__init__(v, v, 0, name)

    @staticmethod
    def calc_schedule(v, *args) -> np.ndarray:
        return np.array([v])

    def to_config(self) -> dict:
        return {"v": self.v, "name": self.name}


@SchedulerRegistry.register("linear")
class LinearScheduler(Scheduler):
    """Linearly increments the KL weight from `v_min` to `v_max`. Reaches `v_max` after
    `max_steps` calls to `step()`"""

    @staticmethod
    def calc_schedule(v_min, v_max, max_steps):
        return np.linspace(v_min, v_max, max_steps + 1)


class CyclicScheduler(LinearScheduler):
    """Linearly increments the weight from `v0` to `v1` over `r * max_steps` and plateuas at
    `v1` from `r * max_steps` to `max_steps`. Every `max_steps` number of calls to `step()`, resets
    the weight to `v0`"""

    def __init__(
        self,
        v_min: float = 0,
        v_max: float = 1,
        max_steps: int = 1000,
        name: str = "v",
        r: float = 0.5,
    ):
        if not 0 <= r <= 1:
            raise ValueError(f"Step fraction 'r' must be in [0, 1]! got: {r:0.2f}")

        super().__init__(v_min, v_max, int(r * max_steps), name)

        self.max_steps = max_steps
        self.r = r

        sched_0 = self.schedule[:-1]
        sched_1 = np.full(self.max_steps - len(sched_0), self.v_max)

        self.schedule = np.concatenate((sched_0, sched_1))

    @property
    def v(self) -> float:
        return self.schedule[self.i % self.max_steps]

    def get_params(self) -> Iterable:
        items = super().get_params()
        items.append(self.r)

        return items


@SchedulerRegistry.register("manual")
class ManualScheduler(Scheduler):
    """Step the weight according to the input schedule

    Parameters
    ----------
    schedule : ArrayLike
        the precise schedule
    """

    def __init__(self, schedule: ArrayLike):
        self.schedule = np.array(schedule)
        self.v_min = schedule[0]
        self.v_max = schedule[-1]
        self.max_steps = len(schedule)

        self.i = 0

    @property
    def v(self) -> float:
        """The current KL weight"""
        return self.schedule[min(len(self) - 1, self.i)]

    @staticmethod
    def calc_schedule(v0, v1, max_steps):
        raise NotImplementedError("`ManualKLScheduler` does not calculate a schedule!")

    @classmethod
    def from_steps_and_weights(cls, steps_weights: Iterable[tuple[int, float]]) -> ManualScheduler:
        """Build a ManualKLScheduler from an implicit schedule defined in terms of the number of steps at a given weight

        Parameters
        ----------
        steps_and_weights : Iterable[tuple[int, float]]
            an iterable containing pairs of the number of steps and the given weight. I.e.,
            steps_weights = [(2, 0.1), (3, 0.2)] would correspond to the schedule
            [0.1, 0.1, 0.2, 0.2, 0.2]
        """
        schedule = np.concatenate([[v] * n for n, v in steps_weights])

        return cls(schedule)

    def to_config(self) -> dict:
        return {"schedule": self.schedule.tolist()}

    def get_params(self):
        return [self.schedule.tolist()]
