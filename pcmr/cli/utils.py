from argparse import ArgumentError, ArgumentParser, Action, Namespace
from typing import Iterable, NamedTuple, Optional, Sequence, Union


class RogiCalculationResult(NamedTuple):
    featurizer: str
    dataset: str
    task: Optional[str]
    n_valid: int
    rogi: float


class DatasetAndTaskAction(Action):
    def __init__(self, *args, choices: Iterable, **kwargs):
        super().__init__(*args, **kwargs)
        self._choices = set(choices)

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence],
        option_string: Optional[str] = None,
    ):
        if isinstance(values, str):
            dataset, task = self.parse_value(values)
            setattr(namespace, self.dest, (dataset, task))
        else:
            datasets_tasks = [self.parse_value(value) for value in values]
            setattr(namespace, self.dest, datasets_tasks)

    def parse_value(self, value: str):
        tokens = value.split("/")

        if len(tokens) == 1:
            dataset = tokens[0]
            task = None
        elif len(tokens) == 2:
            dataset, task = tokens
        else:
            raise ArgumentError(self, f"arguments must of form: ('A', 'A/B')")

        dataset = dataset.upper()
        task = task or None
        if self._choices and dataset not in self._choices:
            raise ArgumentError(
                self, f"Invalid choice: 'dataset' must one of {self._choices}. got: {dataset}"
            )

        return dataset, task
