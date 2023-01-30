from typing import NamedTuple, Optional


class RogiCalculationResult(NamedTuple):
    featurizer: str
    dataset: str
    task: Optional[str]
    n_valid: int
    rogi: float


def dataset_and_task(value) -> tuple[str, Optional[str]]:
    tokens = value.split("/")

    if len(tokens) == 1:
        dataset, task = tokens[0], None
    elif len(tokens) == 2:
        dataset, task = tokens
    else:
        raise ValueError("value must be of form: ('A', 'A/B')")

    dataset = dataset.upper()
    task = task or None

    return dataset, task
