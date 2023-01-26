from typing import Iterable


class InvalidDatasetError(ValueError):
    def __init__(self, dataset: str, datasets: Iterable[str]) -> None:
        super().__init__(f"Invalid dataset! got: '{dataset}'. expected one of: {datasets}.")


class InvalidTaskError(ValueError):
    def __init__(self, task: str, dataset: str, tasks: Iterable[str]) -> None:
        valid_tasks = tasks if len(tasks) > 1 else [*tasks, None]
        super().__init__(
            f"Invalid task! '{task}' is not a valid task for dataset '{dataset}'. "
            f"expected one of: {valid_tasks}"
        )
