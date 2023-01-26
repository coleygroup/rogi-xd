from argparse import ArgumentError, ArgumentParser, Action, Namespace
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from pcmr.data import data
from pcmr.featurizers import FeaturizerRegistry


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
            raise ArgumentError(self, f"invalid choice: 'dataset' must one of {self._choices}")

        return dataset, task


def build_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--featurizers",
        type=lambda s: s.lower(),
        nargs="+",
        choices=FeaturizerRegistry.keys(),
    )
    parser.add_argument(
        "--datasets-tasks",
        "--dt",
        "--datasets",
        action=DatasetAndTaskAction,
        nargs="+",
        choices=data.datasets,
    )
    parser.add_argument("-r", "--repeats", type=int, default=1)
    parser.add_argument("-N", type=int, default=10000, help="the number of data to sumbsample")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "-b", "--batch-size", type=int, help="the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size"
    )

    return parser
