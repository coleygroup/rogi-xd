from typing import Any, Mapping


class PlMixin:
    def _log_split(self, split: str, metrics: Mapping[str, Any], *args, **kwargs):
        self.log_dict({f"{split}/{k}": v for k, v in metrics.items()}, *args, **kwargs)
