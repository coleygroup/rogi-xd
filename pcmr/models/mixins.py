from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import torch


class LoggingMixin:
    def _log_split(self, split: str, metrics: Mapping[str, Any], *args, **kwargs):
        self.log_dict({f"{split}/{k}": v for k, v in metrics.items()}, *args, **kwargs)


class SaveAndLoadMixin:
    def save(self, save_dir: PathLike):
        save_dir = Path(save_dir)
        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p_state_dict)
        p_config.write_text(json.dumps(self.to_config(), indent=2))
    
    @classmethod
    def load(cls, save_dir: PathLike) -> object:
        save_dir = Path(save_dir)
        p_state_dict = save_dir / "model.pt"
        p_config = save_dir / "model.json"

        o = cls.from_config(json.loads(p_config.read_text()))
        o.load_state_dict(torch.load(p_state_dict))

        return o